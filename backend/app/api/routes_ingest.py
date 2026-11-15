"""Document upload, ingestion, status, listing, deletion and chunk provenance.

Role in architecture: the write side of the product. Every document that
enters the system is registered here FIRST (so it has a server-generated
identity, an owner, a tenant and an access policy before any bytes are
parsed), then ingested, then transitioned to a terminal lifecycle state.
Nothing downstream ever infers identity from a filename, and nothing
downstream decides visibility on its own: guardrails/access does.

Two upload paths, same registry and same ingestion:
  * AWS      — presigned S3 POST so bytes go browser -> S3 directly, never
               through Lambda (which bills by duration and caps payloads).
  * local    — single-step multipart, since local dev has no S3.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile

from app.agents import retriever
from app.config import get_settings
from app.documents import registry
from app.guardrails import access
from app.guardrails.auth import resolve_user
from app.guardrails.quota import check_upload_quota
from app.models.schemas import (
    RETRYABLE_ERRORS,
    AccessContext,
    ChunkDetail,
    ClassificationLevel,
    DocumentRecord,
    DocumentStatus,
    DocumentSummary,
    IndexJobResponse,
    PresignedUploadResponse,
)
from app.observability.logging import get_logger
from app.rag import index_store, retriever_snapshot
from app.rag.index_lock import IndexLockTimeout
from app.rag.pdf import PdfExtractionError

router = APIRouter()
_logger = get_logger()

_UPLOAD_EXPIRY_S = 900
_ALLOWED_EXT = {"pdf", "md", "txt"}

# Magic bytes we verify for formats that have them. A .pdf extension is a
# client-supplied claim; the parser will be handed these bytes regardless, so
# the content itself must be checked rather than trusted.
_PDF_MAGIC = b"%PDF-"

_ROLE_RE = re.compile(r"^[a-z0-9_\-]{1,40}$")

Roles = Annotated[
    str | None,
    Query(description="Comma-separated roles that may read this document; empty = owner only"),
]
Level = Annotated[ClassificationLevel, Query(description="Sensitivity level")]


def _ctx(user: dict[str, Any]) -> AccessContext:
    return access.context_for(user)


def _require_aws() -> Any:
    settings = get_settings()
    if settings.local_mode:
        raise HTTPException(status_code=501, detail="uploads require the deployed (AWS) API")
    return settings


def _safe_filename(filename: str) -> str:
    """Validate the extension and sanitize the stem to a key-safe form.

    Sanitizes rather than rejects so real-world names ('FIMP+RFI Draft.pdf')
    work — disallowed characters become underscores. Path separators and '..'
    cannot survive, so the result is never traversal-capable. Idempotent.
    """
    filename = filename.replace("\\", "/").rsplit("/", 1)[-1]  # strip any path
    stem, dot, ext = filename.rpartition(".")
    if not dot or ext.lower() not in _ALLOWED_EXT:
        raise HTTPException(status_code=400, detail="file must be .pdf, .md, or .txt")
    safe_stem = re.sub(r"[^\w.\- ]+", "_", stem).replace("..", "_").strip("_ ") or "document"
    return f"{safe_stem[:100]}.{ext.lower()}"


def _parse_roles(raw: str | None) -> list[str]:
    roles = sorted({r.strip().lower() for r in (raw or "").split(",") if r.strip()})
    for role in roles:
        if not _ROLE_RE.match(role):
            raise HTTPException(status_code=400, detail=f"invalid role name {role!r}")
    return roles


def _check_classification(ctx: AccessContext, level: ClassificationLevel) -> None:
    """Nobody may publish above their own clearance: a PUBLIC-cleared user
    labelling a document RESTRICTED would make it unreadable to themselves and
    is almost certainly a mistake; the reverse (an INTERNAL user downgrading a
    sensitive file to PUBLIC) is the real risk and is the tenant admin's call."""
    if level.rank > ctx.clearance.rank:
        raise HTTPException(
            status_code=403,
            detail=f"your clearance ({ctx.clearance.value}) is below {level.value}",
        )


def _validate_content(filename: str, contents: bytes) -> None:
    """Reject empty files and content that contradicts its extension."""
    if not contents:
        raise HTTPException(status_code=400, detail="file is empty")
    settings = get_settings()
    if len(contents) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"document exceeds the {settings.max_upload_bytes} byte limit",
        )
    if filename.lower().endswith(".pdf") and not contents.startswith(_PDF_MAGIC):
        raise HTTPException(status_code=400, detail="file does not appear to be a valid PDF")


def _content_type_for(filename: str) -> str:
    if filename.lower().endswith(".pdf"):
        return "application/pdf"
    if filename.lower().endswith(".md"):
        return "text/markdown"
    return "text/plain"


def _register(
    *,
    ctx: AccessContext,
    filename: str,
    contents: bytes,
    storage_key: str,
    allowed_roles: list[str],
    classification: ClassificationLevel,
) -> DocumentRecord:
    """Create the document's canonical identity record before any parsing."""
    now = registry.utc_now()
    record = DocumentRecord(
        document_id=registry.new_document_id(),
        owner_id=ctx.user_id,
        tenant_id=ctx.tenant_id,
        allowed_roles=allowed_roles,
        classification=classification,
        original_filename=filename,
        safe_filename=_safe_filename(filename),
        content_type=_content_type_for(filename),
        file_size=len(contents),
        checksum_sha256=hashlib.sha256(contents).hexdigest(),
        storage_key=storage_key,
        status=DocumentStatus.UPLOADED,
        created_at=now,
        updated_at=now,
    )
    return registry.put(record)


def _upload_path(record: DocumentRecord) -> Path:
    """Where local storage keeps an upload's bytes.

    Kept after indexing, not deleted: a dead-lettered document is retried
    from these bytes, the same way the AWS path retries from the S3 object.
    Removed when the document is deleted.
    """
    name = f"{record.document_id}_{record.safe_filename}"
    return Path(get_settings().index_dir) / "uploads" / name


def _ingest_or_422(record: DocumentRecord, local: Path, *, keep: bool = False) -> IndexJobResponse:
    """Run ingestion, translating typed extraction failures into 422s.

    The document's registry state is written by ingest_document() on every
    path, so a 422 here still leaves an inspectable FAILED record with a
    machine-readable error_code rather than a vanished document.
    """
    from app.rag.ingest_runtime import ingest_document

    try:
        chunks_indexed, version = ingest_document(
            local,
            document_id=record.document_id,
            owner_id=record.owner_id,
            source_filename=record.original_filename,
        )
    except (index_store.ConcurrentUpdateError, IndexLockTimeout) as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "error_code": "PUBLISH_CONTENTION",
                "message": "The index was busy; the document is saved and can be retried.",
            },
        ) from exc
    except PdfExtractionError as exc:
        _logger.warning(
            "ingest_rejected",
            extra={"data": f"{record.document_id} {exc.code.value}: {exc.message}"},
        )
        raise HTTPException(
            status_code=422, detail={"error_code": exc.code.value, "message": exc.message}
        ) from exc
    except Exception as exc:
        _logger.error(
            "ingest_failed",
            extra={"data": f"{record.document_id} {type(exc).__name__}: {exc}"},
        )
        raise HTTPException(status_code=500, detail="ingestion failed") from exc
    finally:
        if not keep:
            local.unlink(missing_ok=True)

    retriever.reset_cache()  # so the next query sees the new document
    return IndexJobResponse(
        doc_id=record.document_id, chunks_indexed=chunks_indexed, index_version=version
    )


# ---------------------------------------------------------------- AWS upload flow


@router.post("/documents")
def request_upload(
    filename: str,
    user: dict[str, Any] = Depends(resolve_user),
    roles: Roles = None,
    classification: Level = ClassificationLevel.INTERNAL,
) -> PresignedUploadResponse:
    settings = _require_aws()
    ctx = _ctx(user)
    safe = _safe_filename(filename)
    allowed_roles = _parse_roles(roles)
    _check_classification(ctx, classification)
    check_upload_quota(user, incoming_bytes=0)  # doc-count check before issuing a URL

    import boto3

    now = registry.utc_now()
    document_id = registry.new_document_id()
    key = f"uploads/{ctx.user_id}/{document_id}/{safe}"

    # Registered as UPLOADING before the URL is issued, so an abandoned upload
    # is visible as an incomplete document rather than leaving an orphan object.
    registry.put(
        DocumentRecord(
            document_id=document_id,
            owner_id=ctx.user_id,
            tenant_id=ctx.tenant_id,
            allowed_roles=allowed_roles,
            classification=classification,
            original_filename=filename,
            safe_filename=safe,
            content_type=_content_type_for(safe),
            file_size=0,
            checksum_sha256="",
            storage_key=key,
            status=DocumentStatus.UPLOADING,
            created_at=now,
            updated_at=now,
        )
    )

    s3 = boto3.client("s3", region_name=settings.aws_region)
    post = s3.generate_presigned_post(
        Bucket=settings.s3_bucket_docs,
        Key=key,
        Conditions=[["content-length-range", 1, settings.max_upload_bytes]],
        ExpiresIn=_UPLOAD_EXPIRY_S,
    )
    return PresignedUploadResponse(
        doc_id=document_id,
        upload_url=post["url"],
        fields=post["fields"],
        filename=safe,
        max_bytes=settings.max_upload_bytes,
        expires_in_seconds=_UPLOAD_EXPIRY_S,
    )


@router.post("/documents/{doc_id}/index")
def index_document(doc_id: str, user: dict[str, Any] = Depends(resolve_user)) -> IndexJobResponse:
    """Pull the just-uploaded object and ingest it.

    The S3 key comes from the registry, not from a client-supplied filename:
    the client cannot point this at another user's object.
    """
    settings = _require_aws()
    owner_id = str(user["user_id"])

    record = registry.get(doc_id, owner_id=owner_id)
    if record is None:
        raise HTTPException(status_code=404, detail="document not found")

    import boto3
    from botocore.exceptions import ClientError

    local = Path("/tmp") / f"{record.document_id}_{record.safe_filename}"
    s3 = boto3.client("s3", region_name=settings.aws_region)
    try:
        s3.download_file(settings.s3_bucket_docs, record.storage_key, str(local))
    except ClientError as exc:
        raise HTTPException(status_code=404, detail="uploaded object not found") from exc

    contents = local.read_bytes()
    try:
        _validate_content(record.safe_filename, contents)
    except HTTPException:
        local.unlink(missing_ok=True)
        raise
    check_upload_quota(user, incoming_bytes=len(contents))

    record.file_size = len(contents)
    record.checksum_sha256 = hashlib.sha256(contents).hexdigest()
    record.status = DocumentStatus.UPLOADED
    record.updated_at = registry.utc_now()
    registry.put(record)

    return _ingest_or_422(record, local)


# ---------------------------------------------------------------- local upload flow


@router.post("/documents/local-upload")
def local_upload(
    file: UploadFile,
    user: dict[str, Any] = Depends(resolve_user),
    roles: Roles = None,
    classification: Level = ClassificationLevel.INTERNAL,
) -> IndexJobResponse:
    """Single-step direct upload for local dev and compose: no S3, no presigned URL."""
    settings = get_settings()
    if not settings.local_mode:
        raise HTTPException(status_code=501, detail="use the presigned upload flow in AWS mode")

    ctx = _ctx(user)
    original = file.filename or "document"
    safe = _safe_filename(original)
    allowed_roles = _parse_roles(roles)
    _check_classification(ctx, classification)
    contents = file.file.read()
    _validate_content(safe, contents)
    check_upload_quota(user, incoming_bytes=len(contents))

    record = _register(
        ctx=ctx,
        filename=original,
        contents=contents,
        storage_key=f"local/{ctx.user_id}/{safe}",
        allowed_roles=allowed_roles,
        classification=classification,
    )

    # Written under the document id, so two users uploading the same filename
    # never collide on disk either.
    local = _upload_path(record)
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(contents)

    return _ingest_or_422(record, local, keep=True)


@router.post("/documents/{doc_id}/retry")
def retry_document(doc_id: str, user: dict[str, Any] = Depends(resolve_user)) -> IndexJobResponse:
    """Re-run ingestion for a dead-lettered document from its stored bytes.

    Only for failures a retry can fix (RETRYABLE_ERRORS). A scanned PDF fails
    the same way every time, and offering to retry it would only teach people
    the button does nothing.
    """
    ctx = _ctx(user)
    record = registry.find_visible(doc_id, ctx)
    if record is None or not (record.owner_id == ctx.user_id or ctx.is_admin):
        raise HTTPException(status_code=404, detail="document not found")
    if record.status != DocumentStatus.FAILED or (record.error_code or "") not in RETRYABLE_ERRORS:
        raise HTTPException(status_code=409, detail="this document cannot be retried")

    settings = get_settings()
    if settings.local_mode:
        local = _upload_path(record)
        if not local.exists():
            raise HTTPException(status_code=410, detail="the uploaded bytes are no longer stored")
        return _ingest_or_422(record, local, keep=True)

    import boto3
    from botocore.exceptions import ClientError

    local = Path("/tmp") / f"{record.document_id}_{record.safe_filename}"
    try:
        boto3.client("s3", region_name=settings.aws_region).download_file(
            settings.s3_bucket_docs, record.storage_key, str(local)
        )
    except ClientError as exc:
        raise HTTPException(status_code=410, detail="the uploaded object is gone") from exc
    return _ingest_or_422(record, local)


# ---------------------------------------------------------------- read / delete


@router.get("/documents")
def list_documents(user: dict[str, Any] = Depends(resolve_user)) -> list[DocumentSummary]:
    """Every document the caller may read: their own, plus any shared into
    their tenant by role. Never another tenant's."""
    ctx = _ctx(user)
    return [
        DocumentSummary.from_record(r, viewer_id=ctx.user_id) for r in registry.list_visible(ctx)
    ]


@router.get("/documents/{doc_id}")
def get_document(doc_id: str, user: dict[str, Any] = Depends(resolve_user)) -> DocumentSummary:
    """Lifecycle state for one document, so the client can poll for INDEXED."""
    ctx = _ctx(user)
    record = registry.find_visible(doc_id, ctx)
    if record is None:
        # 404 (not 403) for someone else's document: the caller learns nothing
        # about whether that id exists.
        raise HTTPException(status_code=404, detail="document not found")
    return DocumentSummary.from_record(record, viewer_id=ctx.user_id)


@router.get("/documents/{doc_id}/chunks/{chunk_id}")
def get_chunk(
    doc_id: str, chunk_id: str, user: dict[str, Any] = Depends(resolve_user)
) -> ChunkDetail:
    """One chunk in full, for provenance verification in the source drawer.

    Authorized with the very same rule retrieval uses, so a chunk that could
    never have been cited to this caller cannot be fetched by id either.
    """
    ctx = _ctx(user)
    record = registry.find_visible(doc_id, ctx)
    snap = retriever_snapshot.current()
    chunk = next((c for c in snap.chunks if c.chunk_id == chunk_id and c.doc_id == doc_id), None)
    if chunk is None or not access.can_see(ctx, chunk):
        raise HTTPException(status_code=404, detail="chunk not found")
    if record is None and chunk.owner_id:
        # A per-user document whose registry row the caller cannot see.
        raise HTTPException(status_code=404, detail="chunk not found")
    return ChunkDetail(
        chunk_id=chunk.chunk_id,
        document_id=chunk.doc_id,
        source_filename=chunk.source_filename,
        section_path=chunk.section_path,
        page_number=chunk.page_number,
        char_start=chunk.char_start,
        char_end=chunk.char_end,
        text=chunk.text,
        provenance_hash=chunk.provenance_hash,
        # The checksum the hash was computed with, from the chunk itself: a
        # registry row can be re-uploaded, and the corpus has none at all.
        document_checksum_sha256=chunk.document_checksum
        or (record.checksum_sha256 if record else ""),
        classification=chunk.classification,
    )


@router.delete("/documents/{doc_id}")
def delete_document(doc_id: str, user: dict[str, Any] = Depends(resolve_user)) -> DocumentSummary:
    """Purge a document's chunks from the index and tombstone the record.

    Only the owner or a tenant admin may delete; a reader who was merely
    granted a role on the document gets the same 404 as a stranger.
    """
    from app.rag.ingest_runtime import delete_document as purge

    ctx = _ctx(user)
    record = registry.find_visible(doc_id, ctx)
    if record is None or not (record.owner_id == ctx.user_id or ctx.is_admin):
        raise HTTPException(status_code=404, detail="document not found")

    purge(doc_id, record.owner_id)
    retriever.reset_cache()  # deleted content must stop being retrievable now
    if get_settings().local_mode:
        _upload_path(record).unlink(missing_ok=True)
    updated = registry.get(doc_id, owner_id=record.owner_id)
    return DocumentSummary.from_record(updated or record, viewer_id=ctx.user_id)
