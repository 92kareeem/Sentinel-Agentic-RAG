"""LangGraph state schema — single source of truth for what flows between nodes.

Role in architecture: every node reads and returns this exact shape. Keeping
it a flat TypedDict (not nested Pydantic) is a LangGraph requirement — the
graph diffs/merges state between node invocations.
"""

from typing import Literal, NotRequired, TypedDict

from app.models.schemas import Chunk, Citation, CriticScores, RefusalReason

# Imported eagerly, not under TYPE_CHECKING: LangGraph resolves this
# TypedDict's annotations at runtime to build its state schema, so a string
# forward reference fails there. There is no import cycle to avoid —
# observability.tracing imports nothing from app.agents (the original
# `trace: object` annotation cited a cycle that does not exist, and cost the
# whole agent package its type checking).
from app.observability.tracing import TraceRecorder


class AgentState(TypedDict):
    # What the user actually asked. Set once and never reassigned — the answer
    # the user reads must address THIS, whatever retrieval did along the way.
    query: str
    # What retrieval is currently searching with. Starts equal to `query` and
    # is replaced by repair_rewrite with a keyword-dense reformulation.
    #
    # These used to be one field. repair_rewrite overwrote it, so after a
    # rewrite the synthesizer answered the reformulation and the critic scored
    # relevance against it too — the whole pipeline agreed on an answer to a
    # question nobody asked. For "can we refund after 20 days if onboarding
    # started?", a rewrite to "annual subscription refund window" produces a
    # confident, correctly-cited answer that silently drops the onboarding
    # exception: the worst failure shape for this product, because nothing
    # about it looks wrong.
    search_query: str
    user_id: str
    doc_id: str | None  # scope retrieval to one uploaded document; None = whole index
    trace: TraceRecorder
    attempt: int
    model: str
    token_budget_left: int
    deadline_ts: float
    retrieved: list[Chunk]
    answer: str
    citations: list[Citation]
    critic: CriticScores | None
    status: Literal["running", "answered", "refused"]
    # Set alongside status="refused" so the API can tell the user WHY it
    # declined. None while running or answered.
    refusal_reason: RefusalReason | None
    conversation_history: list[dict[str, str]]
    # The draft that failed the critic or the grounding gate, kept for
    # diagnosis only. Deliberately NOT part of any API response: it is exactly
    # the text the system just decided it could not stand behind.
    rejected_draft: NotRequired[str]
