"""Document upload, ingestion, status, listing and deletion.

Role in architecture: the write side of the product. Every document that
enters the system is registered here FIRST (so it has a server-generated
identity and an owner before any bytes are parsed), then ingested, then
transitioned to a terminal lifecycle state. Nothing downstream ever infers
identity from a filename.

Two upload paths, same registry and same ingestion:
  * AWS      — presigned S3 POST so bytes go browser -> S3 directly, never
               through Lambda (which bills by duration and caps payloads).
  * local    — single-step multipart, since local dev has no S3.
"""

import hashlib
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, UploadFile

from app.agents import retriever
from app.config import get_settings
from app.documents import registry
from app.guardrails.auth import resolve_user
from app.guardrails.quota import check_upload_quota, record_upload
from app.models.schemas import (
    DocumentRecord,
    DocumentStatus,
    DocumentSummary,
    IndexJobResponse,
    PresignedUploadResponse,
)
from app.observability.logging import get_logger
from app.rag.pdf import PdfExtractionError

router = APIRouter()
_logger = get_logger()

_UPLOAD_EXPIRY_S = 900
_ALLOWED_EXT = {"pdf", "md", "txt"}

# Magic bytes we verify for formats that have them. A .pdf extension is a
# client-supplied claim; the parser will be handed these bytes regardless, so
# the content itself must be checked rather than trusted.
_PDF_MAGIC = b"%PDF-"


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
        raise HTTPException(
            status_code=400, detail="file does not appear to be a valid PDF"
        )


def _content_type_for(filename: str) -> str:
    if filename.lower().endswith(".pdf"):
        return "application/pdf"
    if filename.lower().endswith(".md"):
        return "text/markdown"
    return "text/plain"


def _register(
    *, owner_id: str, filename: str, contents: bytes, storage_key: str
) -> DocumentRecord:
    """Create the document's canonical identity record before any parsing."""
    now = registry.utc_now()
    record = DocumentRecord(
        document_id=registry.new_document_id(),
        owner_id=owner_id,
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


def _ingest_or_422(record: DocumentRecord, local: Path) -> IndexJobResponse:
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
        local.unlink(missing_ok=True)

    retriever.reset_cache()  # so the next query sees the new document
    return IndexJobResponse(
        doc_id=record.document_id, chunks_indexed=chunks_indexed, index_version=version
    )


# ---------------------------------------------------------------- AWS upload flow


@router.post("/documents")
def request_upload(
    filename: str, user: dict[str, Any] = Depends(resolve_user)
) -> PresignedUploadResponse:
    settings = _require_aws()
    safe = _safe_filename(filename)
    check_upload_quota(user, incoming_bytes=0)  # doc-count check before issuing a URL

    import boto3

    now = registry.utc_now()
    document_id = registry.new_document_id()
    key = f"uploads/{user['user_id']}/{document_id}/{safe}"

    # Registered as UPLOADING before the URL is issued, so an abandoned upload
    # is visible as an incomplete document rather than leaving an orphan object.
    registry.put(
        DocumentRecord(
            document_id=document_id,
            owner_id=str(user["user_id"]),
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
def index_document(
    doc_id: str, user: dict[str, Any] = Depends(resolve_user)
) -> IndexJobResponse:
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

    result = _ingest_or_422(record, local)
    record_upload(user, len(contents))
    return result


# ---------------------------------------------------------------- local upload flow


@router.post("/documents/local-upload")
def local_upload(
    file: UploadFile, user: dict[str, Any] = Depends(resolve_user)
) -> IndexJobResponse:
    """Single-step direct upload for local dev: no S3, no presigned URL."""
    settings = get_settings()
    if not settings.local_mode:
        raise HTTPException(status_code=501, detail="use the presigned upload flow in AWS mode")

    original = file.filename or "document"
    safe = _safe_filename(original)
    contents = file.file.read()
    _validate_content(safe, contents)
    check_upload_quota(user, incoming_bytes=len(contents))

    record = _register(
        owner_id=str(user["user_id"]),
        filename=original,
        contents=contents,
        storage_key=f"local/{user['user_id']}/{safe}",
    )

    # Written under the document id, so two users uploading the same filename
    # never collide on disk either.
    local = Path(settings.index_dir) / f"{record.document_id}_{safe}"
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(contents)

    result = _ingest_or_422(record, local)
    record_upload(user, len(contents))
    return result


# ---------------------------------------------------------------- read / delete


@router.get("/documents")
def list_documents(user: dict[str, Any] = Depends(resolve_user)) -> list[DocumentSummary]:
    """Every document belonging to the caller. Never another tenant's."""
    return [
        DocumentSummary.from_record(r) for r in registry.list_for_owner(str(user["user_id"]))
    ]


@router.get("/documents/{doc_id}")
def get_document(
    doc_id: str, user: dict[str, Any] = Depends(resolve_user)
) -> DocumentSummary:
    """Lifecycle state for one document, so the client can poll for INDEXED."""
    record = registry.get(doc_id, owner_id=str(user["user_id"]))
    if record is None:
        # 404 (not 403) for someone else's document: the caller learns nothing
        # about whether that id exists.
        raise HTTPException(status_code=404, detail="document not found")
    return DocumentSummary.from_record(record)


@router.delete("/documents/{doc_id}")
def delete_document(
    doc_id: str, user: dict[str, Any] = Depends(resolve_user)
) -> DocumentSummary:
    """Purge a document's chunks from the index and tombstone the record."""
    from app.rag.ingest_runtime import delete_document as purge

    owner_id = str(user["user_id"])
    record = registry.get(doc_id, owner_id=owner_id)
    if record is None:
        raise HTTPException(status_code=404, detail="document not found")

    purge(doc_id, owner_id)
    retriever.reset_cache()  # deleted content must stop being retrievable now
    updated = registry.get(doc_id, owner_id=owner_id)
    return DocumentSummary.from_record(updated or record)
