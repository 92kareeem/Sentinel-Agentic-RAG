"""Document registry: the authoritative record of what documents exist,
who owns them, and what state they are in.

Role in architecture: before this existed, document state had to be inferred
from the FAISS index — which cannot express "uploaded but not yet parsed",
"failed because the PDF was encrypted", or "owned by user A". Ownership in
particular was unrepresentable, so document-scoped queries could not be
authorized. The registry is now the single source of truth for identity and
lifecycle; the index is a derived artifact.

Two backends, one interface:
  * local_mode  -> a JSON file written atomically (os.replace) under index_dir
  * AWS         -> DynamoDB, PK=USER#<owner_id> SK=DOC#<document_id>

The DynamoDB key layout answers the two access patterns we actually have
("list this user's documents" = query by PK; "fetch this one document" = get
by PK+SK) without a scan or a secondary index.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.models.schemas import AccessContext, DocumentErrorCode, DocumentRecord, DocumentStatus

_LOCAL_FILE = "documents.json"

# Guards the local JSON file against concurrent read-modify-write from
# multiple request threads in the same process. AWS mode does not use this —
# DynamoDB item writes are atomic per item.
_local_lock = threading.RLock()


def new_document_id() -> str:
    """Server-generated, collision-free document identity.

    Deliberately NOT derived from the filename: two users can upload
    "policy.pdf", one user can upload it twice, and neither should collide
    or silently overwrite the other in the index.
    """
    return uuid.uuid4().hex


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _age_seconds(created_at: str) -> float:
    """Seconds since a record was created; 0.0 if the timestamp is unparseable
    (a malformed row should never be mistaken for an ancient one)."""
    try:
        created = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except (ValueError, TypeError):
        return 0.0
    return (datetime.now(UTC) - created).total_seconds()


def _expire_if_abandoned(record: DocumentRecord) -> DocumentRecord:
    """Report a long-dead UPLOADING record as FAILED.

    A document is registered as UPLOADING *before* its presigned URL is issued,
    so an upload the user abandons leaves a row that never advances. Those rows
    were previously immortal: they sat in the sidebar forever and — because the
    upload quota counted them — eventually locked the user out of uploading at
    all.

    This is computed on read rather than written back. A read path that mutates
    would turn every document listing into a write, and the stored row is still
    the honest record of what happened; only its PRESENTATION needs to admit
    the upload is never coming. A durable reaper can come later without
    changing this behaviour.
    """
    if record.status != DocumentStatus.UPLOADING:
        return record
    if _age_seconds(record.created_at) < get_settings().upload_abandon_seconds:
        return record
    return record.model_copy(
        update={
            "status": DocumentStatus.FAILED,
            "error_code": DocumentErrorCode.UPLOAD_ABANDONED,
            "error_message": "Upload was never completed.",
        }
    )


# ---------------------------------------------------------------- local backend


def _local_path() -> Path:
    return Path(get_settings().index_dir) / _LOCAL_FILE


def _local_read_all() -> dict[str, dict[str, Any]]:
    path = _local_path()
    if not path.exists():
        return {}
    try:
        data: dict[str, dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
        return data
    except (json.JSONDecodeError, OSError):
        # A truncated registry is recoverable state, not a reason to 500 every
        # request: treat it as empty and let subsequent writes rebuild it.
        return {}


def _local_write_all(data: dict[str, dict[str, Any]]) -> None:
    path = _local_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # write-then-rename: a reader never observes a half-written registry
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------- dynamo backend


def _table() -> Any:
    import boto3

    settings = get_settings()
    return boto3.resource("dynamodb", region_name=settings.aws_region).Table(
        settings.ddb_table_documents
    )


def _ddb_keys(owner_id: str, document_id: str) -> dict[str, str]:
    return {"pk": f"USER#{owner_id}", "sk": f"DOC#{document_id}"}


# ---------------------------------------------------------------- public API


class InvalidTransition(RuntimeError):
    """A lifecycle change the state machine does not allow, or one that lost a
    race: the stored status was no longer the one this writer read."""


# The document lifecycle as an explicit state machine. Every write through
# update_status() is checked against it, and in DynamoDB the check is a
# conditional write on the status the writer read, so two concurrent writers
# (an index job and a delete, say) cannot both win. DELETED is terminal: a
# tombstoned document can never be resurrected by a late ingestion finishing.
_TRANSITIONS: dict[DocumentStatus, frozenset[DocumentStatus]] = {
    DocumentStatus.UPLOADING: frozenset({
        DocumentStatus.UPLOADED, DocumentStatus.PROCESSING,
        DocumentStatus.FAILED, DocumentStatus.DELETED,
    }),
    DocumentStatus.UPLOADED: frozenset({
        DocumentStatus.PROCESSING, DocumentStatus.FAILED, DocumentStatus.DELETED,
    }),
    DocumentStatus.PROCESSING: frozenset({
        DocumentStatus.INDEXED, DocumentStatus.FAILED, DocumentStatus.DELETED,
    }),
    DocumentStatus.INDEXED: frozenset({
        DocumentStatus.PROCESSING, DocumentStatus.INDEXED, DocumentStatus.DELETED,
    }),
    DocumentStatus.FAILED: frozenset({
        DocumentStatus.PROCESSING, DocumentStatus.FAILED, DocumentStatus.DELETED,
    }),
    DocumentStatus.DELETED: frozenset(),
}


def can_transition(current: DocumentStatus, target: DocumentStatus) -> bool:
    return target in _TRANSITIONS[current]


def put(record: DocumentRecord, *, expected_status: DocumentStatus | None = None) -> DocumentRecord:
    """Create or overwrite a document record.

    With expected_status, the write succeeds only if the stored record still
    has that status — a compare-and-set, atomic at DynamoDB and under the
    process lock locally. Raises InvalidTransition when it lost the race.
    """
    if get_settings().local_mode:
        with _local_lock:
            data = _local_read_all()
            if expected_status is not None:
                stored = data.get(record.document_id, {}).get("status")
                if stored != expected_status.value:
                    raise InvalidTransition(
                        f"{record.document_id}: expected {expected_status.value}, found {stored}"
                    )
            data[record.document_id] = record.model_dump(mode="json")
            _local_write_all(data)
        return record

    item = {**_ddb_keys(record.owner_id, record.document_id), **record.model_dump(mode="json")}
    if expected_status is None:
        _table().put_item(Item=item)
        return record
    from botocore.exceptions import ClientError

    try:
        _table().put_item(
            Item=item,
            ConditionExpression="#s = :expected",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={":expected": expected_status.value},
        )
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "ConditionalCheckFailedException":
            raise InvalidTransition(
                f"{record.document_id}: status changed concurrently"
            ) from exc
        raise
    return record


def get(document_id: str, *, owner_id: str | None = None) -> DocumentRecord | None:
    """Fetch one document. When owner_id is given, a document owned by anyone
    else reads as absent — callers cannot distinguish "not yours" from "does
    not exist", which is the correct behavior for an authorization boundary
    (it leaks no information about other tenants' document ids).
    """
    if get_settings().local_mode:
        with _local_lock:
            raw = _local_read_all().get(document_id)
        if raw is None:
            return None
        record = DocumentRecord(**raw)
    else:
        if owner_id is None:
            # DynamoDB needs the full key; a global lookup would require a GSI
            # we deliberately do not maintain. Every real caller knows the owner.
            raise ValueError("owner_id is required to fetch a document in AWS mode")
        resp = _table().get_item(Key=_ddb_keys(owner_id, document_id))
        item = resp.get("Item")
        if item is None:
            return None
        record = DocumentRecord(**{k: v for k, v in item.items() if k not in ("pk", "sk")})

    if owner_id is not None and record.owner_id != owner_id:
        return None
    return _expire_if_abandoned(record)


def list_for_owner(owner_id: str) -> list[DocumentRecord]:
    """All non-deleted documents belonging to one user, newest first."""
    if get_settings().local_mode:
        with _local_lock:
            raw = _local_read_all()
        records = [DocumentRecord(**v) for v in raw.values()]
        records = [r for r in records if r.owner_id == owner_id]
    else:
        from boto3.dynamodb.conditions import Key

        resp = _table().query(KeyConditionExpression=Key("pk").eq(f"USER#{owner_id}"))
        records = [
            DocumentRecord(**{k: v for k, v in item.items() if k not in ("pk", "sk")})
            for item in resp.get("Items", [])
        ]

    records = [_expire_if_abandoned(r) for r in records if r.status != DocumentStatus.DELETED]
    return sorted(records, key=lambda r: r.created_at, reverse=True)


def list_for_tenant(tenant_id: str) -> list[DocumentRecord]:
    """Every non-deleted document in one tenant, newest first.

    Local mode filters the registry file. AWS mode is a filtered Scan: the
    table is keyed by owner for the two hot paths (list mine, fetch mine), and
    a tenant-wide listing is an admin-and-sharing path that runs rarely over a
    table sized in the hundreds. A GSI would be the right call at volume and
    is deliberately not provisioned on the free tier until that volume exists.
    """
    if not tenant_id:
        return []
    if get_settings().local_mode:
        with _local_lock:
            raw = _local_read_all()
        records = [DocumentRecord(**v) for v in raw.values()]
        records = [r for r in records if r.tenant_id == tenant_id]
    else:
        from boto3.dynamodb.conditions import Attr

        resp = _table().scan(FilterExpression=Attr("tenant_id").eq(tenant_id))
        records = [
            DocumentRecord(**{k: v for k, v in item.items() if k not in ("pk", "sk")})
            for item in resp.get("Items", [])
        ]
    records = [_expire_if_abandoned(r) for r in records if r.status != DocumentStatus.DELETED]
    return sorted(records, key=lambda r: r.created_at, reverse=True)


def find_visible(document_id: str, ctx: AccessContext) -> DocumentRecord | None:
    """One document the caller may see, or None (absent and forbidden alike).

    Tries the caller's own partition first, which is the common case and a
    single key read; falls back to the tenant listing for documents shared in
    by role. Visibility itself is decided by guardrails/access, nowhere else.
    """
    from app.guardrails import access

    record = get(document_id, owner_id=ctx.user_id)
    if record is None and ctx.tenant_id:
        record = next(
            (r for r in list_for_tenant(ctx.tenant_id) if r.document_id == document_id), None
        )
    return record if access.document_visible(ctx, record) else None


def list_visible(ctx: AccessContext) -> list[DocumentRecord]:
    """Documents the caller owns plus those shared with them in the tenant."""
    from app.guardrails import access

    mine = {r.document_id: r for r in list_for_owner(ctx.user_id)}
    for r in list_for_tenant(ctx.tenant_id):
        if r.document_id not in mine and access.document_visible(ctx, r):
            mine[r.document_id] = r
    return sorted(mine.values(), key=lambda r: r.created_at, reverse=True)


def update_status(
    document_id: str,
    owner_id: str,
    status: DocumentStatus,
    *,
    chunk_count: int | None = None,
    page_count: int | None = None,
    index_version: str | None = None,
    parser_version: str | None = None,
    embedding_model: str | None = None,
    error_code: DocumentErrorCode | None = None,
    error_message: str | None = None,
) -> DocumentRecord | None:
    """Advance a document's lifecycle state, recording why if it failed.

    Checked against the lifecycle state machine and written as a
    compare-and-set on the status that was read. Raises InvalidTransition for
    an illegal move (e.g. anything out of DELETED) or a lost race.
    """
    record = get(document_id, owner_id=owner_id)
    if record is None:
        return None
    previous = record.status
    if not can_transition(previous, status):
        raise InvalidTransition(f"{document_id}: {previous.value} -> {status.value} is not allowed")

    record.status = status
    if status == DocumentStatus.PROCESSING:
        record.ingest_attempts += 1
    record.updated_at = utc_now()
    if chunk_count is not None:
        record.chunk_count = chunk_count
    if page_count is not None:
        record.page_count = page_count
    if index_version is not None:
        record.index_version = index_version
    if parser_version is not None:
        record.parser_version = parser_version
    if embedding_model is not None:
        record.embedding_model = embedding_model
    # Clearing on success matters: a document that failed, was re-uploaded and
    # succeeded must not keep showing a stale error in the UI.
    record.error_code = error_code
    record.error_message = error_message
    return put(record, expected_status=previous)


def mark_deleted(document_id: str, owner_id: str) -> DocumentRecord | None:
    return update_status(document_id, owner_id, DocumentStatus.DELETED)


def reset_local_registry() -> None:
    """Test helper: drop the local registry file."""
    if not get_settings().local_mode:
        raise RuntimeError("reset_local_registry is local_mode only")
    with _local_lock:
        path = _local_path()
        if path.exists():
            path.unlink()
