"""POST /v1/query — the full guardrail chain, then the agent graph.

Role in architecture: the only place the guardrail order is wired
(auth -> validate -> injection -> pii -> quota -> cost governor), so an
auditor reads one function to see every gate a query passes through.
"""

import time
from typing import Any

from fastapi import APIRouter, Depends

from app.agents.graph import build_graph
from app.agents.state import AgentState
from app.config import get_settings
from app.guardrails import cost_governor, injection, input_validation, pii, quota
from app.guardrails.auth import resolve_user
from app.llm.groq_client import CircuitOpenError
from app.models.schemas import (
    Citation,
    CriticScores,
    QueryRequest,
    QueryResponse,
    RefusalResponse,
    TokenUsage,
)
from app.observability.tracing import TraceRecorder, put_trace

router = APIRouter()

_graph = None  # compiled once per process, reused across warm invocations


def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


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

    settings = get_settings()
    trace = TraceRecorder(user_id=str(user["user_id"]), query_redacted=scrubbed)
    state: AgentState = {
        "query": scrubbed,
        "user_id": str(user["user_id"]),
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
    }
    t0 = time.perf_counter()
    try:
        result = _get_graph().invoke(state)
    except CircuitOpenError as exc:
        from fastapi import HTTPException

        put_trace(trace.to_dict("error"))
        raise HTTPException(status_code=503, detail="LLM circuit breaker open") from exc

    latency_ms = int((time.perf_counter() - t0) * 1000)
    refused = result["status"] == "refused" or result["answer"].strip() == "INSUFFICIENT_CONTEXT"
    put_trace(trace.to_dict("refused" if refused else "answered"))

    if refused:
        return RefusalResponse(
            trace_id=trace.trace_id,
            reason=result["answer"],
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
