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
from app.models.schemas import DocumentErrorCode, DocumentRecord, DocumentStatus

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


def put(record: DocumentRecord) -> DocumentRecord:
    """Create or overwrite a document record."""
    if get_settings().local_mode:
        with _local_lock:
            data = _local_read_all()
            data[record.document_id] = record.model_dump(mode="json")
            _local_write_all(data)
        return record

    item = {**_ddb_keys(record.owner_id, record.document_id), **record.model_dump(mode="json")}
    _table().put_item(Item=item)
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
    return record


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

    records = [r for r in records if r.status != DocumentStatus.DELETED]
    return sorted(records, key=lambda r: r.created_at, reverse=True)


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
    """Advance a document's lifecycle state, recording why if it failed."""
    record = get(document_id, owner_id=owner_id)
    if record is None:
        return None

    record.status = status
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
    return put(record)


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
