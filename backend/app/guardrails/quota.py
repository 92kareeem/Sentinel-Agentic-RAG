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

# local_mode fallback: in-memory counters, same semantics, laptop only
_local_counts: dict[str, int] = {}
_local_uploads: dict[str, dict[str, int]] = {}  # user_id -> {count, bytes}


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


def check_upload_quota(user: dict[str, Any], incoming_bytes: int) -> None:
    """Reject (429) if this upload would exceed the user's lifetime doc/byte
    limits. Read-then-check (uploads are rare — no concurrency concern like
    queries have). Admin bypasses."""
    if user.get("is_admin"):
        return
    doc_limit = int(user.get("upload_doc_limit", 10))
    byte_limit = int(user.get("upload_bytes_limit", 20_971_520))
    uid = str(user["user_id"])
    settings = get_settings()

    if settings.local_mode:
        cur = _local_uploads.get(uid, {"count": 0, "bytes": 0})
    else:
        import boto3

        table = boto3.resource("dynamodb", region_name=settings.aws_region).Table(
            settings.ddb_table_quotas
        )
        item = table.get_item(Key={"quota_key": f"{uid}#U"}).get("Item", {})
        cur = {"count": int(item.get("count", 0)), "bytes": int(item.get("bytes", 0))}

    if cur["count"] >= doc_limit:
        raise HTTPException(status_code=429, detail=f"upload limit reached ({doc_limit} documents)")
    if cur["bytes"] + incoming_bytes > byte_limit:
        raise HTTPException(
            status_code=429, detail=f"upload storage limit reached ({byte_limit} bytes)"
        )


def record_upload(user: dict[str, Any], size_bytes: int) -> None:
    """Count a successful upload against the user's lifetime totals."""
    if user.get("is_admin"):
        return
    uid = str(user["user_id"])
    settings = get_settings()
    if settings.local_mode:
        cur = _local_uploads.setdefault(uid, {"count": 0, "bytes": 0})
        cur["count"] += 1
        cur["bytes"] += size_bytes
        return

    import boto3

    table = boto3.resource("dynamodb", region_name=settings.aws_region).Table(
        settings.ddb_table_quotas
    )
    table.update_item(
        Key={"quota_key": f"{uid}#U"},
        UpdateExpression="ADD #c :one, #b :sz",
        ExpressionAttributeNames={"#c": "count", "#b": "bytes"},
        ExpressionAttributeValues={":one": 1, ":sz": size_bytes},
    )
