"""API-key auth: hash the header, resolve the user, cache briefly.

Role in architecture: second auth layer behind API Gateway's edge key check.
The edge stops unauthenticated floods; this layer resolves *identity*
(user_id, is_admin, limits) for quotas and trace ownership. Keys are stored
only as sha256 hashes — a leaked table leaks nothing usable.
"""

import hashlib
import time
from typing import Any

from fastapi import Header, HTTPException

from app.config import get_settings

_CACHE_TTL_S = 60.0
_cache: dict[str, tuple[dict[str, Any], float]] = {}

# local_mode fixture users (no DynamoDB on a laptop)
_LOCAL_USERS = {
    "demo-local": {"user_id": "demo", "is_admin": False, "daily_query_limit": 50},
    "admin-local": {"user_id": "admin", "is_admin": True, "daily_query_limit": 0},
}


def hash_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()


def _lookup_user(key_hash: str) -> dict[str, Any] | None:
    import boto3
    from boto3.dynamodb.conditions import Key

    settings = get_settings()
    table = boto3.resource("dynamodb", region_name=settings.aws_region).Table(
        settings.ddb_table_users
    )
    resp = table.query(
        IndexName="api_key_hash-index",
        KeyConditionExpression=Key("api_key_hash").eq(key_hash),
        Limit=1,
    )
    items = resp.get("Items", [])
    return items[0] if items else None


def resolve_user(x_api_key: str = Header(...)) -> dict[str, Any]:
    """FastAPI dependency: x-api-key -> user record, else 401."""
    settings = get_settings()
    if settings.local_mode:
        user = _LOCAL_USERS.get(x_api_key)
        if user is None:
            raise HTTPException(status_code=401, detail="invalid API key")
        return user

    key_hash = hash_key(x_api_key)
    cached = _cache.get(key_hash)
    if cached and time.monotonic() - cached[1] < _CACHE_TTL_S:
        return cached[0]

    user = _lookup_user(key_hash)
    if user is None:
        raise HTTPException(status_code=401, detail="invalid API key")
    _cache[key_hash] = (user, time.monotonic())
    return user
