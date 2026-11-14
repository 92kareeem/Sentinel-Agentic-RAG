"""The control plane API: telemetry, knowledge gaps, the ratchet, traces.

Role in architecture: the endpoints written for whoever OWNS the system
rather than whoever asks it questions. Everything else in this API answers
"what does my corpus say?"; these answer "how is it performing, what is the
corpus missing, and which fixed mistakes are locked in?"

Scoping: a regular user sees their own activity. A tenant admin sees the
whole tenant. Nobody sees another tenant — the questions people ask are at
least as sensitive as the documents they ask about.

All of it is computed on read. A scheduled job needs something to run it,
which on a zero-cost deployment means paying to stay awake with nobody
looking. Reading on demand costs nothing when nobody asks.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from app.config import get_settings
from app.guardrails import access
from app.guardrails.auth import require_admin, resolve_user
from app.learning import casebook, ratchet
from app.models.schemas import (
    AccessContext,
    KnowledgeGapReport,
    PromoteRequest,
    RatchetCandidate,
    RatchetItemView,
    RatchetView,
    RetireRequest,
    TelemetryReport,
    TraceSummary,
)
from app.observability import telemetry
from app.observability.tracing import list_traces

router = APIRouter()

_TRACE_SCAN_LIMIT = 2_000


def _ctx(user: dict[str, Any]) -> AccessContext:
    return access.context_for(user)


def _scoped_traces(ctx: AccessContext, limit: int) -> list[dict[str, Any]]:
    """The caller's traces, or the whole tenant's for a tenant admin."""
    if ctx.is_admin and ctx.tenant_id:
        return [
            r
            for r in list_traces(limit=limit)
            if r.get("tenant_id") == ctx.tenant_id or r.get("user_id") == ctx.user_id
        ]
    return list_traces(limit=limit, user_id=ctx.user_id)


# ----------------------------------------------------------------- telemetry


@router.get("/insights/telemetry", response_model=TelemetryReport)
def get_telemetry(
    window_hours: int = Query(default=24, ge=1, le=24 * 30),
    user: dict[str, Any] = Depends(resolve_user),
) -> TelemetryReport:
    """Latency percentiles, token burn, estimated cost, cache and refusal
    rates, and a per-node cost profile for the window."""
    ctx = _ctx(user)
    return telemetry.build_report(_scoped_traces(ctx, _TRACE_SCAN_LIMIT), window_hours=window_hours)


@router.get("/insights/traces", response_model=list[TraceSummary])
def recent_traces(
    limit: int = Query(default=25, ge=1, le=200),
    user: dict[str, Any] = Depends(resolve_user),
) -> list[TraceSummary]:
    """The trace browser: most recent requests first. Full records come from
    GET /v1/traces/{trace_id}, which applies the same ownership rule."""
    ctx = _ctx(user)
    rows = _scoped_traces(ctx, limit)
    rows = sorted(rows, key=lambda r: str(r.get("created_at", "")), reverse=True)[:limit]
    out: list[TraceSummary] = []
    for r in rows:
        out.append(
            TraceSummary(
                trace_id=str(r["trace_id"]),
                created_at=str(r.get("created_at", "")),
                user_id=str(r.get("user_id", "")),
                query_redacted=str(r.get("query_redacted", "")),
                final_status=str(r.get("final_status", "")),
                latency_ms=int(float(r.get("latency_ms", 0) or 0)),
                total_tokens=int(float(r.get("total_tokens", 0) or 0)),
                est_cost_usd=float(r.get("est_cost_usd", 0) or 0),
                repair_count=int(float(r.get("repair_count", 0) or 0)),
                cache_hit=r.get("cache_hit") in (True, "True", "true"),
            )
        )
    return out


# ------------------------------------------------------------ knowledge gaps


@router.get("/insights/knowledge-gaps", response_model=KnowledgeGapReport)
def knowledge_gaps(
    window_days: int = Query(default=30, ge=1, le=365),
    user: dict[str, Any] = Depends(resolve_user),
) -> KnowledgeGapReport:
    """Questions this workspace could not answer, clustered and ranked
    (ADR 0007). The document owner's work list."""
    return casebook.knowledge_gaps(str(user["user_id"]), window_days=window_days)


# --------------------------------------------------------------- the ratchet


def _writable(ctx: AccessContext) -> bool:
    """The suite is a file in the repository. Only a server running next to
    the repository (local storage) can write it, and only for an admin."""
    return get_settings().local_mode and ctx.is_admin


def _promote_command(case_id: str, owner: str) -> str:
    return f"python evals/promote.py promote {case_id} --owner {owner}"


@router.get("/insights/ratchet", response_model=RatchetView)
def get_ratchet(user: dict[str, Any] = Depends(resolve_user)) -> RatchetView:
    """The regression suite and the cases waiting to be reviewed into it."""
    ctx = _ctx(user)
    suite = ratchet.load(get_settings().regressions_path)
    cases = (
        casebook.list_for_tenant(ctx.tenant_id, fallback_owner=ctx.user_id)
        if ctx.is_admin
        else casebook.list_for_owner(ctx.user_id)
    )
    candidates = [
        RatchetCandidate(
            case_id=c.case_id,
            question=c.question,
            diagnosis=c.diagnosis,
            outcome=c.outcome,
            correction=c.correction,
            asked_at=c.created_at,
            owner_id=c.owner_id,
            promote_command=_promote_command(c.case_id, c.owner_id),
        )
        for c in ratchet.candidates(cases, suite)[:100]
    ]
    items = [
        RatchetItemView(
            id=i.id,
            question=i.question,
            expect=i.expect,
            status=i.status,
            reference_answer=i.reference_answer,
            note=i.note,
            diagnosis=i.source.diagnosis.value,
            promoted_at=i.promoted_at,
            locked_at=i.locked_at,
        )
        for i in suite.active
    ]
    return RatchetView(
        enforced=sum(i.status == "enforced" for i in suite.active),
        pending=sum(i.status == "pending" for i in suite.active),
        retired=len(suite.retired),
        items=items,
        candidates=candidates,
        writable=_writable(ctx),
    )


@router.post("/insights/ratchet/promote", response_model=RatchetItemView)
def promote_case(
    req: PromoteRequest, user: dict[str, Any] = Depends(resolve_user)
) -> RatchetItemView:
    """Turn a reviewed case into a PENDING regression (ADR 0008).

    Human-only by construction: admin key, an explicit click, and for an
    `answer` test either the reviewer's reference or the reader's correction.
    It lands pending, so it never blocks CI until an eval run shows it pass.
    """
    require_admin(user)
    ctx = _ctx(user)
    if not _writable(ctx):
        raise HTTPException(
            status_code=409,
            detail="the regression suite lives in git; promote from a checkout with "
            + _promote_command(req.case_id, ctx.user_id),
        )
    path = get_settings().regressions_path
    suite = ratchet.load(path)
    case = next(
        (
            c
            for c in casebook.list_for_tenant(ctx.tenant_id, fallback_owner=ctx.user_id)
            if c.case_id == req.case_id
        ),
        None,
    )
    if case is None:
        raise HTTPException(status_code=404, detail="case not found")
    try:
        item = ratchet.promote(
            suite,
            case,
            expect="refuse" if req.expect == "refuse" else "answer",
            reference=req.reference,
            note=req.note,
        )
        ratchet.save(suite, path)
    except ratchet.PromotionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return RatchetItemView(
        id=item.id,
        question=item.question,
        expect=item.expect,
        status=item.status,
        reference_answer=item.reference_answer,
        note=item.note,
        diagnosis=item.source.diagnosis.value,
        promoted_at=item.promoted_at,
        locked_at=item.locked_at,
    )


@router.post("/insights/ratchet/{item_id}/retire")
def retire_item(
    item_id: str, req: RetireRequest, user: dict[str, Any] = Depends(resolve_user)
) -> dict[str, str]:
    """Retire a regression with a reason. Nothing is ever deleted (ADR 0008)."""
    require_admin(user)
    if not _writable(_ctx(user)):
        raise HTTPException(
            status_code=409,
            detail="retire from a checkout: "
            f'python evals/promote.py retire {item_id} --reason "..."',
        )
    path = get_settings().regressions_path
    suite = ratchet.load(path)
    try:
        retired = ratchet.retire(suite, item_id, req.reason)
        ratchet.save(suite, path)
    except ratchet.PromotionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"id": retired.id, "retired_at": retired.retired_at, "reason": retired.reason}
