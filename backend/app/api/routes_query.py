"""POST /v1/query — the full guardrail chain, then the agent graph.

Role in architecture: the only place the guardrail order is wired
(auth -> validate -> injection -> pii -> quota -> cost governor), so an
auditor reads one function to see every gate a query passes through.
"""

import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from app.agents.graph import build_graph
from app.agents.state import AgentState
from app.config import get_settings
from app.guardrails import cost_governor, injection, input_validation, pii, quota
from app.guardrails.auth import resolve_user
from app.learning import casebook
from app.llm.groq_client import CircuitOpenError
from app.models.schemas import (
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

router = APIRouter()
_logger = get_logger()

_graph = None  # compiled once per process, reused across warm invocations


def _get_graph() -> Any:
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def _repaired_citation_count(trace: TraceRecorder) -> int:
    """How many citations the grounding gate had to re-point, from the trace.

    Read back off the trace rather than threaded through AgentState: the graph
    already records it, and adding a field to the state schema for something
    only the API reports would make every node's contract wider for no reason.
    """
    total = 0
    for step in trace.steps:
        if step["name"] == "grounding_check":
            try:
                total += int(step["meta"].get("repaired_citations", 0))
            except (TypeError, ValueError):
                continue
    return total


def _persist_observations(
    trace: TraceRecorder,
    result: Any,
    refused: bool,
    user_id: str,
    question: str,
    doc_id: str | None,
) -> None:
    """Write the trace, the volume counter and (if warranted) a case.

    All of it is best-effort, and deliberately AFTER the answer is fully
    determined. A user who asked a hard question must not lose their response
    because the record of it could not be stored — and the casebook exists
    precisely to make hard questions better, so failing them would invert the
    feature's purpose. put_trace previously ran unguarded here, so a DynamoDB
    blip turned a finished answer into a 500.
    """
    outcome = CaseOutcome.REFUSED if refused else CaseOutcome.ANSWERED
    try:
        put_trace(trace.to_dict("refused" if refused else "answered"))
    except Exception as exc:  # noqa: BLE001
        _logger.warning(
            "trace_write_failed",
            extra={"trace_id": trace.trace_id, "data": f"{type(exc).__name__}: {exc}"},
        )

    casebook.bump_totals(user_id, outcome)

    reason = result.get("refusal_reason")
    repaired = _repaired_citation_count(trace)
    diagnosis = casebook.diagnose(
        outcome=outcome,
        refusal_reason=str(reason) if reason else None,
        retrieved_chunks=len(result["retrieved"]),
        repaired_citations=repaired,
    )
    if diagnosis is None:  # a clean answer — nothing to learn from
        return
    casebook.record(
        Case(
            case_id=casebook.new_case_id(),
            owner_id=user_id,
            trace_id=trace.trace_id,
            question=question,
            outcome=outcome,
            diagnosis=diagnosis,
            refusal_reason=reason,
            doc_id=doc_id,
            retrieved_chunks=len(result["retrieved"]),
            repaired_citations=repaired,
            repair_count=trace.repair_count,
            created_at=casebook.utc_now(),
        )
    )


@router.post("/query", response_model=None)
def query(
    req: QueryRequest,
    user: dict[str, Any] = Depends(resolve_user),          # 1. 401
) -> QueryResponse | RefusalResponse:
    input_validation.validate_query(req)                    # 2. 400/413
    injection.screen_query(req.query)                       # 3. 422
    scrubbed = pii.scrub(req.query)                         # 4. before logs/Groq
    quota.check_quota(user)                                 # 5. 429
    budget = cost_governor.allocate_budget()                # 6. hard caps

    # 7. document scope authorization. A doc_id is a client-supplied string;
    # without this check a caller could scope a query to another tenant's
    # document id and read its content back through the answer. Retrieval also
    # filters by owner (defense in depth), but the request is rejected here so
    # the caller gets a clear 404 rather than a confusing empty refusal.
    if req.doc_id is not None:
        from app.documents import registry
        from app.models.schemas import DocumentStatus

        record = registry.get(req.doc_id, owner_id=str(user["user_id"]))
        if record is None:
            raise HTTPException(status_code=404, detail="document not found")
        if record.status != DocumentStatus.INDEXED:
            raise HTTPException(
                status_code=409,
                detail=f"document is not queryable yet (status: {record.status.value})",
            )

    settings = get_settings()
    trace = TraceRecorder(user_id=str(user["user_id"]), query_redacted=scrubbed)
    history = [
        {"role": turn.role, "content": turn.content}
        for turn in req.conversation_history
    ]
    state: AgentState = {
        "query": scrubbed,
        # Retrieval starts from the user's own wording; repair_rewrite may
        # replace this, but never `query`.
        "search_query": scrubbed,
        "user_id": str(user["user_id"]),
        "doc_id": req.doc_id,
        "trace": trace,
        "attempt": 0,
        "model": settings.groq_model_simple,
        "token_budget_left": budget.token_budget,
        "deadline_ts": budget.deadline_ts,
        "retrieved": [],
        "answer": "",
        "citations": [],
        "critic": None,
        "status": "running",
        "refusal_reason": None,
        "conversation_history": history,
    }
    t0 = time.perf_counter()
    try:
        result = _get_graph().invoke(state)
    except CircuitOpenError as exc:
        _logger.warning("circuit_open", extra={"trace_id": trace.trace_id, "data": str(exc)})
        put_trace(trace.to_dict("error"))
        raise HTTPException(status_code=503, detail="LLM circuit breaker open") from exc
    except Exception as exc:
        # Any node can raise if Groq is unreachable/misbehaving in a way the
        # retry loop can't fix (e.g. a deterministic 4xx, not a transient
        # 429/5xx) — groq_client re-raises as RuntimeError once retries are
        # exhausted. Without this, that exception was uncaught here and
        # crashed the request with a raw 500 instead of a clean, retryable
        # response — a single bad Groq call would take the whole request down
        # with it instead of degrading gracefully like every other failure
        # mode in this graph. Logged with the exception type/message BEFORE
        # converting to a clean client response — otherwise the real cause is
        # invisible server-side, all you'd ever see is "upstream LLM error".
        _logger.error(
            "query_failed",
            extra={"trace_id": trace.trace_id, "data": f"{type(exc).__name__}: {exc}"},
        )
        put_trace(trace.to_dict("error"))
        raise HTTPException(status_code=503, detail="upstream LLM error — please retry") from exc

    latency_ms = int((time.perf_counter() - t0) * 1000)
    refused = result["status"] == "refused" or is_insufficient_context(result["answer"])
    try:
        _persist_observations(trace, result, refused, str(user["user_id"]), scrubbed, req.doc_id)
    except Exception as exc:  # noqa: BLE001
        # The guarantee lives HERE, not in each writer, so it holds however
        # many things this grows to record. An answer has already been
        # computed and paid for; nothing about filing it away is worth
        # turning that into a 500 for the user.
        _logger.warning(
            "observations_write_failed",
            extra={"trace_id": trace.trace_id, "data": f"{type(exc).__name__}: {exc}"},
        )

    if refused:
        # Falls back to INSUFFICIENT_EVIDENCE for the one path that reaches
        # here without passing through refusal_node: the synthesizer returning
        # the sentinel with status still "running".
        reason_code = result.get("refusal_reason") or RefusalReason.INSUFFICIENT_EVIDENCE
        return RefusalResponse(
            trace_id=trace.trace_id,
            # Derived from the code, NOT read from result["answer"]. On a
            # critic or grounding failure that field held the rejected draft,
            # so the API handed back the very text the graph had just refused
            # to stand behind. refusal_node now overwrites it too; generating
            # the prose here as well means no future node can reintroduce the
            # leak by leaving model output in `answer`.
            reason=safe_refusal_text(reason_code),
            reason_code=reason_code,
            best_effort_context=[c.chunk_id for c in result["retrieved"]],
        )
    return QueryResponse(
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
        latency_ms=latency_ms,
    )
