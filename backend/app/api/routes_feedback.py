"""POST /v1/feedback — a reader's verdict on an answer they were given.

Role in architecture: the only input to the casebook that does not come from
the graph. Every other case is the system noticing its own failure — a
refusal, a grounding rejection, a repaired citation. That leaves one class of
failure permanently invisible: an answer that passed every gate, shipped with
citations, and was wrong anyway. If any upstream check could detect it, that
check would already have refused. Only the person who read it can say so.

That is also why this endpoint matters for the eval ratchet and not just for
the report. A refusal case says a question failed; a correction says what
passing looks like. The second is the raw material for a regression test.
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from app.guardrails import pii
from app.guardrails.auth import resolve_user
from app.learning import casebook
from app.models.schemas import (
    Case,
    CaseOutcome,
    FeedbackRequest,
    FeedbackResponse,
)
from app.observability.tracing import get_trace

router = APIRouter()

# A trace that never produced an answer has nothing to rate.
_RATEABLE = {"answered": CaseOutcome.ANSWERED, "refused": CaseOutcome.REFUSED}


@router.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(
    req: FeedbackRequest,
    user: dict[str, Any] = Depends(resolve_user),
) -> FeedbackResponse:
    """Record a verdict on one answer. Idempotent per reader and answer.

    The question is read back from the trace rather than accepted from the
    client. Otherwise a caller could attach any question they liked to a real
    answer, and — because corrections feed the eval ratchet — plant a "known
    correct" answer in the regression suite.

    Refusals are rateable, not just answers. "You refused, but the handbook
    covers this" is a false refusal reported by the one person positioned to
    notice, and a correction attached to it is exactly the test that stops
    the system refusing that question again.

    Traces expire after 30 days, and in local mode live only in memory, so
    rating an answer from before a restart returns 404. Both are the right
    behaviour: a verdict must be tied to a trace this system can still show.
    """
    owner_id = str(user["user_id"])
    trace = get_trace(req.trace_id)

    # 404, not 403, for somebody else's trace. A 403 would confirm that the id
    # exists, and trace ids are the handle on another tenant's questions.
    if trace is None or str(trace.get("user_id")) != owner_id:
        raise HTTPException(status_code=404, detail="answer not found")

    outcome = _RATEABLE.get(str(trace.get("final_status")))
    if outcome is None:
        raise HTTPException(
            status_code=409, detail="this request did not produce an answer to rate"
        )

    asked_at = str(trace["created_at"])
    case: Case | None = None
    diagnosis = casebook.VERDICT_DIAGNOSIS.get(req.verdict)
    if diagnosis is not None:
        correction = pii.scrub(req.correction).strip() if req.correction else None
        case = Case(
            case_id=casebook.feedback_case_id(req.trace_id),
            owner_id=owner_id,
            trace_id=req.trace_id,
            # Already PII-scrubbed: it is the redacted query the trace stored.
            question=str(trace.get("query_redacted", "")),
            outcome=outcome,
            diagnosis=diagnosis,
            repair_count=int(trace.get("repair_count", 0)),
            # When the question was ASKED — see record_feedback() for why the
            # window follows the question, not the click.
            created_at=asked_at,
            correction=correction or None,
        )

    recorded = casebook.record_feedback(
        owner_id=owner_id,
        trace_id=req.trace_id,
        verdict=req.verdict,
        day=asked_at[:10],
        case=case,
    )
    return FeedbackResponse(
        trace_id=req.trace_id,
        verdict=req.verdict,
        recorded=recorded,
        # Only reported when this call opened it. On a repeat the reader's
        # earlier verdict stands, and it may not have opened a case at all.
        case_id=case.case_id if (recorded and case is not None) else None,
    )
