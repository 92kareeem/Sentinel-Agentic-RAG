"""The query pipeline, shared by the JSON route, the SSE route and the demo.

Role in architecture: three entry points answer questions, and they must not
drift apart on the things that matter — guardrail order, authorization, what
a refusal may contain, what gets recorded. So there is one implementation:

    prepare()   guardrails, scope authorization, budget, trace, initial state
    run()       the agent graph, as an iterator of (node, state) updates
    finalize()  the response, built from the result by one function, plus
                trace / casebook / cache bookkeeping that can never fail it

Guardrail order (an auditor reads this one function to see every gate):
auth -> validate -> injection -> pii -> quota -> cost governor -> scope.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException

from app.agents.graph import build_graph
from app.agents.state import AgentState
from app.config import get_settings
from app.guardrails import access, cost_governor, injection, input_validation, pii, quota
from app.learning import casebook
from app.models.schemas import (
    AccessContext,
    Case,
    CaseOutcome,
    Citation,
    CriticScores,
    QueryRequest,
    QueryResponse,
    RefusalReason,
    RefusalResponse,
    TokenUsage,
    is_insufficient_context,
    safe_refusal_text,
)
from app.observability.logging import get_logger
from app.observability.tracing import TraceRecorder, put_trace
from app.rag import answer_cache, retriever_snapshot

_logger = get_logger()

_graph: Any = None  # compiled once per process, reused across warm invocations


def default_graph() -> Any:
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


@dataclass
class Prepared:
    req: QueryRequest
    ctx: AccessContext
    scrubbed: str
    trace: TraceRecorder
    state: AgentState
    cache_key: answer_cache.CacheKey | None


def prepare(
    req: QueryRequest,
    user: dict[str, Any],
    *,
    enforce_quota: bool = True,
    token_budget: int | None = None,
) -> Prepared:
    """Every gate a question passes before the graph sees it. Raises
    HTTPException with the status that gate owns; nothing here is a refusal."""
    input_validation.validate_query(req)                    # 400 / 413
    injection.screen_query(req.query)                       # 422
    scrubbed = pii.scrub(req.query)                         # before logs or any model
    if enforce_quota:
        quota.check_quota(user)                             # 429
    budget = cost_governor.allocate_budget()                # hard caps
    ctx = access.context_for(user)

    # A doc_id is a client-supplied string; without this check a caller could
    # scope a query to another tenant's document and read it back through the
    # answer. Retrieval also filters before scoring (defense in depth), but
    # the request is rejected here so the caller gets a clear 404 rather than
    # a confusing empty refusal.
    if req.doc_id is not None:
        from app.documents import registry
        from app.models.schemas import DocumentStatus

        record = registry.find_visible(req.doc_id, ctx)
        if record is None:
            raise HTTPException(status_code=404, detail="document not found")
        if record.status != DocumentStatus.INDEXED:
            raise HTTPException(
                status_code=409,
                detail=f"document is not queryable yet (status: {record.status.value})",
            )

    settings = get_settings()
    trace = TraceRecorder(user_id=ctx.user_id, tenant_id=ctx.tenant_id, query_redacted=scrubbed)
    history = [{"role": t.role, "content": t.content} for t in req.conversation_history]
    state: AgentState = {
        "query": scrubbed,
        # Retrieval starts from the user's own wording; repair_rewrite may
        # replace this, but never `query` (ADR 0006).
        "search_query": scrubbed,
        "user_id": ctx.user_id,
        "access": ctx,
        "doc_id": req.doc_id,
        "trace": trace,
        "attempt": 0,
        "model": settings.model_simple,
        "token_budget_left": token_budget if token_budget is not None else budget.token_budget,
        "deadline_ts": budget.deadline_ts,
        "retrieved": [],
        "answer": "",
        "citations": [],
        "critic": None,
        "status": "running",
        "refusal_reason": None,
        "conversation_history": history,
    }

    cache_key = None
    if not history:
        cache_key = answer_cache.make_key(
            question=scrubbed,
            ctx=ctx,
            doc_id=req.doc_id,
            index_version=retriever_snapshot.current().version,
        )
    return Prepared(req, ctx, scrubbed, trace, state, cache_key)


# ------------------------------------------------------------------ running


def iterate(graph: Any, state: AgentState) -> Iterator[tuple[str | None, dict[str, Any]]]:
    """Yield (node_name, state_so_far) after every node, then (None, final).

    Uses LangGraph's combined updates+values stream so the caller learns WHICH
    node finished and also holds the complete state at the end. Falls back to
    invoke() for graph stand-ins that do not stream (tests, older fakes).
    """
    if not hasattr(graph, "stream"):
        yield None, graph.invoke(state)
        return
    final: dict[str, Any] = dict(state)
    for mode, chunk in graph.stream(state, stream_mode=["updates", "values"]):
        if mode == "values":
            final = chunk
        else:
            for node in chunk:
                yield node, final
    yield None, final


def run(prepared: Prepared, graph: Any = None) -> dict[str, Any]:
    """Run to completion. Provider failures become a clean 503."""
    graph = graph or default_graph()
    result: dict[str, Any] = {}
    try:
        for node, state in iterate(graph, prepared.state):
            if node is None:
                result = state
    except Exception as exc:
        raise upstream_failure(prepared, exc) from exc
    return result


def upstream_failure(prepared: Prepared, exc: Exception) -> HTTPException:
    """Log the real cause, store the error trace, and give the client a
    retryable 503 rather than a raw 500 — a bad provider call must degrade
    like every other failure mode in this graph."""
    from app.llm.base import CircuitOpenError

    event = "circuit_open" if isinstance(exc, CircuitOpenError) else "query_failed"
    _logger.error(event, extra={"trace_id": prepared.trace.trace_id,
                                "data": f"{type(exc).__name__}: {exc}"})
    # The 503 matters more than the record of it.
    with contextlib.suppress(Exception):
        put_trace(prepared.trace.to_dict("error"))
    detail = (
        "LLM circuit breaker open"
        if isinstance(exc, CircuitOpenError)
        else "upstream LLM error — please retry"
    )
    return HTTPException(status_code=503, detail=detail)


# ------------------------------------------------------------------ cache


def from_cache(prepared: Prepared) -> QueryResponse | None:
    """A previously verified answer, re-issued under THIS request's trace so
    feedback, telemetry and the trace inspector all work on it unchanged."""
    cached = answer_cache.lookup(prepared.cache_key)
    if cached is None:
        return None
    trace = prepared.trace
    trace.cache_hit = True
    trace.model_path.append(cached.model_used)
    trace.record_step("cache", 0, hit=True, model=cached.model_used)
    response = cached.model_copy(
        update={
            "trace_id": trace.trace_id,
            "repair_count": 0,
            "tokens": TokenUsage(tokens_in=0, tokens_out=0),
            "latency_ms": trace.elapsed_ms(),
        }
    )
    _record(prepared, outcome=CaseOutcome.ANSWERED, result=None, final_status="answered")
    return response


# ------------------------------------------------------------------ finishing


def finalize(prepared: Prepared, result: Mapping[str, Any]) -> QueryResponse | RefusalResponse:
    """The one place a response is built from a graph result."""
    trace = prepared.trace
    refused = result["status"] == "refused" or is_insufficient_context(result["answer"])

    response: QueryResponse | RefusalResponse
    if refused:
        # Falls back to INSUFFICIENT_EVIDENCE for the one path that reaches
        # here without passing through refusal_node: the synthesizer returning
        # the sentinel with status still "running".
        reason_code = result.get("refusal_reason") or RefusalReason.INSUFFICIENT_EVIDENCE
        response = RefusalResponse(
            trace_id=trace.trace_id,
            # Derived from the code, NEVER from result["answer"] (ADR 0005).
            # On a critic or grounding failure that field held the rejected
            # draft; no model output can reach this field.
            reason=safe_refusal_text(reason_code),
            reason_code=reason_code,
            best_effort_context=[c.chunk_id for c in result["retrieved"]],
        )
    else:
        response = QueryResponse(
            trace_id=trace.trace_id,
            answer=result["answer"],
            citations=[Citation(**c.model_dump()) for c in result["citations"]],
            critic=result["critic"] or CriticScores(faithfulness=0.0, relevance=0.0),
            repair_count=trace.repair_count,
            model_used=result["model"],
            tokens=TokenUsage(
                tokens_in=sum(s["tokens_in"] for s in trace.steps),
                tokens_out=sum(s["tokens_out"] for s in trace.steps),
            ),
            latency_ms=trace.elapsed_ms(),
        )
        # Only an answer that went through the critic and the grounding gate
        # is cached. A stub graph that never graded it is not "verified".
        if result.get("critic") is not None:
            answer_cache.store(prepared.cache_key, response)

    _record(
        prepared,
        outcome=CaseOutcome.REFUSED if refused else CaseOutcome.ANSWERED,
        result=result,
        final_status="refused" if refused else "answered",
    )
    return response


def _repaired_citation_count(trace: TraceRecorder) -> int:
    total = 0
    for step in trace.steps:
        if step["name"] == "grounding_check":
            try:
                total += int(step["meta"].get("repaired_citations", 0))
            except (TypeError, ValueError):
                continue
    return total


def _record(
    prepared: Prepared,
    *,
    outcome: CaseOutcome,
    result: Mapping[str, Any] | None,
    final_status: str,
) -> None:
    """Trace, volume counter and (if warranted) a case. Never raises.

    The guarantee lives here, not in each writer, so it holds however many
    things this grows to record: an answer has already been computed and paid
    for, and nothing about filing it away is worth turning that into a 500.
    """
    trace = prepared.trace
    ctx = prepared.ctx
    try:
        put_trace(trace.to_dict(final_status))
    except Exception as exc:  # noqa: BLE001
        _logger.warning(
            "trace_write_failed",
            extra={"trace_id": trace.trace_id, "data": f"{type(exc).__name__}: {exc}"},
        )
    try:
        casebook.bump_totals(ctx.user_id, outcome)
        if result is None:
            return
        reason = result.get("refusal_reason")
        repaired = _repaired_citation_count(trace)
        diagnosis = casebook.diagnose(
            outcome=outcome,
            refusal_reason=str(reason) if reason else None,
            retrieved_chunks=len(result["retrieved"]),
            repaired_citations=repaired,
        )
        if diagnosis is None:
            return
        casebook.record(
            Case(
                case_id=casebook.new_case_id(),
                owner_id=ctx.user_id,
                tenant_id=ctx.tenant_id,
                trace_id=trace.trace_id,
                question=prepared.scrubbed,
                outcome=outcome,
                diagnosis=diagnosis,
                refusal_reason=reason,
                doc_id=prepared.req.doc_id,
                retrieved_chunks=len(result["retrieved"]),
                repaired_citations=repaired,
                repair_count=trace.repair_count,
                created_at=casebook.utc_now(),
            )
        )
    except Exception as exc:  # noqa: BLE001
        _logger.warning(
            "observations_write_failed",
            extra={"trace_id": trace.trace_id, "data": f"{type(exc).__name__}: {exc}"},
        )


# ------------------------------------------------------------------ streaming

# Trace meta that is safe to show while the request is still in flight. This
# is an allow-list on purpose. The critic's `why` and the rewrite's
# `new_query` are model output that can quote or paraphrase a draft that has
# not been verified, and ADR 0005 / ADR 0010 say no unverified text reaches
# the client before the grounding gate has passed it.
SAFE_META = frozenset({
    "label", "source", "model", "downgraded", "chunks", "scope", "rows_excluded",
    "index_version", "ok", "stripped_ratio", "repaired_citations", "reason", "hit",
    "rejected_draft_chars",
})

STAGE_LABEL: dict[str, str] = {
    "router": "Routing",
    "retriever": "Retrieving from the snapshot",
    "synthesizer": "Drafting an answer",
    "critic": "Judging faithfulness",
    "grounding_check": "Verifying grounding",
    "repair_rewrite": "Repairing: rewriting the search",
    "repair_escalate": "Repairing: escalating the model",
    "refusal": "Refusing safely",
    "cache": "Serving a verified answer",
}


def safe_step(step: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": step["name"],
        "started_ms": step.get("started_ms", 0),
        "duration_ms": step.get("duration_ms", 0),
        "tokens_in": step.get("tokens_in", 0),
        "tokens_out": step.get("tokens_out", 0),
        "meta": {k: v for k, v in (step.get("meta") or {}).items() if k in SAFE_META},
    }


def stage_event(node: str, trace: TraceRecorder, state: Mapping[str, Any]) -> dict[str, Any]:
    step = next((s for s in reversed(trace.steps) if s["name"] == node), None)
    return {
        "node": node,
        "label": STAGE_LABEL.get(node, node),
        "attempt": int(state.get("attempt", 0)),
        "step": safe_step(step) if step else None,
    }


GraphFactory = Callable[[], Any]
