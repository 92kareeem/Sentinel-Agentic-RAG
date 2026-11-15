"""API-key auth: hash the header, resolve the user, cache briefly.

Role in architecture: second auth layer behind the edge. The edge stops
unauthenticated floods; this layer resolves *identity* — user, tenant, roles,
clearance, limits — for authorization, quotas and trace ownership. Keys are
stored only as sha256 hashes, so a leaked table leaks nothing usable.

The user record is the input to guardrails/access.context_for(). Fields:

  user_id            stable identity; owns documents, traces and cases
  tenant_id          the organisation; documents never cross it
  roles              strings matched against a document's allowed_roles
  clearance          ClassificationLevel the caller may read up to
  is_admin           tenant admin: sees every document in the tenant,
                     bypasses quotas, may promote regressions
  daily_query_limit  0 = unlimited
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

from fastapi import Header, HTTPException

from app.config import get_settings

_CACHE_TTL_S = 60.0
_cache: dict[str, tuple[dict[str, Any], float]] = {}

# local_mode fixture users (no DynamoDB on a laptop). Two tenants and four
# roles, so tenant isolation, role sharing and clearance can all be exercised
# from a browser with nothing but a different API key.
_LOCAL_USERS: dict[str, dict[str, Any]] = {
    "demo-local": {
        "user_id": "demo",
        "tenant_id": "acme",
        "roles": ["employee"],
        "clearance": "INTERNAL",
        "is_admin": False,
        "daily_query_limit": 50,
    },
    "admin-local": {
        "user_id": "admin",
        "tenant_id": "acme",
        "roles": ["employee", "admin"],
        "clearance": "RESTRICTED",
        "is_admin": True,
        "daily_query_limit": 0,
    },
    "finance-local": {
        "user_id": "finance",
        "tenant_id": "acme",
        "roles": ["employee", "finance"],
        "clearance": "CONFIDENTIAL",
        "is_admin": False,
        "daily_query_limit": 50,
    },
    "contractor-local": {
        "user_id": "contractor",
        "tenant_id": "acme",
        "roles": ["contractor"],
        "clearance": "PUBLIC",
        "is_admin": False,
        "daily_query_limit": 50,
    },
    "globex-local": {
        "user_id": "globex-user",
        "tenant_id": "globex",
        "roles": ["employee"],
        "clearance": "INTERNAL",
        "is_admin": False,
        "daily_query_limit": 50,
    },
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
        return dict(user)

    key_hash = hash_key(x_api_key)
    cached = _cache.get(key_hash)
    if cached and time.monotonic() - cached[1] < _CACHE_TTL_S:
        return cached[0]

    user = _lookup_user(key_hash)
    if user is None:
        raise HTTPException(status_code=401, detail="invalid API key")
    _cache[key_hash] = (user, time.monotonic())
    return user


def require_admin(user: dict[str, Any]) -> dict[str, Any]:
    if not user.get("is_admin"):
        raise HTTPException(status_code=403, detail="admin only")
    return user
