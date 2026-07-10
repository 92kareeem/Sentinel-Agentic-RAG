"""GET /v1/traces/{trace_id} (owner or admin) and GET /v1/traces (admin)."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from app.guardrails.auth import resolve_user
from app.observability.tracing import get_trace, list_traces

router = APIRouter()


@router.get("/traces/{trace_id}")
def get_one(trace_id: str, user: dict[str, Any] = Depends(resolve_user)) -> dict:
    record = get_trace(trace_id)
    if record is None:
        raise HTTPException(status_code=404, detail="trace not found")
    if not user.get("is_admin") and record.get("user_id") != user["user_id"]:
        raise HTTPException(status_code=403, detail="not your trace")
    return record


@router.get("/traces")
def get_many(limit: int = 20, user: dict[str, Any] = Depends(resolve_user)) -> list[dict]:
    if not user.get("is_admin"):
        raise HTTPException(status_code=403, detail="admin only")
    return list_traces(limit=min(limit, 100))
