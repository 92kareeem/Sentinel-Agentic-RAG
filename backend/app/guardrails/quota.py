"""Per-user daily query quota via one atomic DynamoDB conditional update.

Role in architecture: the write IS the check — `ADD count :one` guarded by
`attribute_not_exists(count) OR count < :limit`. DynamoDB evaluates the
condition at write time, so concurrent requests cannot both sneak past a
stale read (there is no read). Failure -> ConditionalCheckFailedException
-> 429 with Retry-After.
"""

import time
from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException

from app.config import get_settings
from app.documents import registry
from app.models.schemas import DocumentStatus

# local_mode fallback: in-memory counter, same semantics, laptop only.
# Upload usage needs no equivalent — it is derived from the registry, which
# already has a local backend (see upload_usage).
_local_counts: dict[str, int] = {}


def _quota_key(user_id: str) -> str:
    return f"{user_id}#Q#{datetime.now(UTC).strftime('%Y-%m-%d')}"


def _seconds_until_utc_midnight() -> int:
    now = datetime.now(UTC)
    return 86400 - (now.hour * 3600 + now.minute * 60 + now.second)


def check_quota(user: dict[str, Any]) -> None:
    """Increment-and-check; raises 429 when the daily limit is hit. Admin bypasses."""
    if user.get("is_admin"):
        return

    settings = get_settings()
    limit = int(user.get("daily_query_limit", 50))
    key = _quota_key(str(user["user_id"]))

    if settings.local_mode:
        _local_counts[key] = _local_counts.get(key, 0) + 1
        if _local_counts[key] > limit:
            raise HTTPException(
                status_code=429,
                detail="daily query quota exceeded",
                headers={"Retry-After": str(_seconds_until_utc_midnight())},
            )
        return

    import boto3
    from botocore.exceptions import ClientError

    table = boto3.resource("dynamodb", region_name=settings.aws_region).Table(
        settings.ddb_table_quotas
    )
    try:
        table.update_item(
            Key={"quota_key": key},
            UpdateExpression="ADD #c :one SET #ttl = if_not_exists(#ttl, :ttl)",
            ConditionExpression="attribute_not_exists(#c) OR #c < :limit",
            ExpressionAttributeNames={"#c": "count", "#ttl": "ttl"},
            ExpressionAttributeValues={
                ":one": 1,
                ":limit": limit,
                ":ttl": int(time.time()) + 7 * 86400,
            },
        )
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "ConditionalCheckFailedException":
            raise HTTPException(
                status_code=429,
                detail="daily query quota exceeded",
                headers={"Retry-After": str(_seconds_until_utc_midnight())},
            ) from exc
        raise


# Statuses that occupy capacity. FAILED is excluded on purpose: a document
# that never indexed stores nothing and answers nothing, and counting it would
# let a user lock themselves out permanently by uploading a few corrupt PDFs.
_LIVE_STATUSES = frozenset(
    {
        DocumentStatus.UPLOADING,
        DocumentStatus.UPLOADED,
        DocumentStatus.PROCESSING,
        DocumentStatus.INDEXED,
    }
)


def upload_usage(owner_id: str) -> tuple[int, int]:
    """(documents, bytes) the owner currently holds, derived from the registry.

    DERIVED, NOT COUNTED — and that is the whole point.

    This used to be a stored counter in the quotas table, incremented on every
    upload and decremented never. It drifted from reality in three ways at
    once: deleting a document did not give the capacity back, so a user who
    uploaded and deleted ten files could never upload again; abandoned
    UPLOADING rows counted forever; and because the counter lived in a
    different table from the documents, recreating the documents table left a
    counter measuring documents that no longer existed. That is exactly what
    happened in production — the counter read 10/10 while the table held zero
    rows.

    The registry is already the source of truth for what a user owns, and
    listing it is a single partition query. Deriving the number cannot drift,
    because there is no second copy to drift from.
    """
    records = registry.list_for_owner(owner_id)  # excludes DELETED, expires abandoned
    live = [r for r in records if r.status in _LIVE_STATUSES]
    return len(live), sum(r.file_size for r in live)


def check_upload_quota(user: dict[str, Any], incoming_bytes: int) -> None:
    """Reject (429) if this upload would exceed the owner's capacity limits.

    Note this is a CAPACITY limit ("how much may you hold at once"), not a
    lifetime one ("how much may you ever upload"). Deleting a document frees
    the capacity immediately, which is what a user expects from a limit
    expressed in documents and bytes. Abuse of the churn that allows is
    covered by the per-day query quota above.
    """
    if user.get("is_admin"):
        return
    doc_limit = int(user.get("upload_doc_limit", 50))
    byte_limit = int(user.get("upload_bytes_limit", 200_000_000))

    count, used_bytes = upload_usage(str(user["user_id"]))

    if count >= doc_limit:
        raise HTTPException(status_code=429, detail=f"upload limit reached ({doc_limit} documents)")
    if used_bytes + incoming_bytes > byte_limit:
        raise HTTPException(
            status_code=429, detail=f"upload storage limit reached ({byte_limit} bytes)"
        )
