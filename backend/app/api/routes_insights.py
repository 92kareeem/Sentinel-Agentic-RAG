"""GET /v1/insights/knowledge-gaps — what the documents could not answer.

Role in architecture: the one endpoint written for the document OWNER rather
than the person asking questions. Everything else in this API answers "what
does my corpus say?"; this answers "what is my corpus missing?", which is the
question a business actually pays to have answered.

Scoped to the caller for the same reason retrieval is: the questions people
ask are at least as sensitive as the documents they ask about, and a gap
report is a very compact summary of what an organisation does not know.
"""

from typing import Any

from fastapi import APIRouter, Depends, Query

from app.guardrails.auth import resolve_user
from app.learning import casebook
from app.models.schemas import KnowledgeGapReport

router = APIRouter()


@router.get("/insights/knowledge-gaps", response_model=KnowledgeGapReport)
def knowledge_gaps(
    window_days: int = Query(default=30, ge=1, le=365),
    user: dict[str, Any] = Depends(resolve_user),
) -> KnowledgeGapReport:
    """Questions this workspace could not answer, clustered and ranked.

    Computed on read rather than materialised on a schedule. At this volume
    the clustering is milliseconds over a single-partition query, and a
    scheduled job would need something to run it — which on a zero-cost
    deployment means either a always-on process or an EventBridge rule
    invoking a Lambda whether or not anyone will look at the result. Reading
    on demand costs nothing when nobody asks.
    """
    return casebook.knowledge_gaps(str(user["user_id"]), window_days=window_days)
