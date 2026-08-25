"""LangGraph state schema — single source of truth for what flows between nodes.

Role in architecture: every node reads and returns this exact shape. Keeping
it a flat TypedDict (not nested Pydantic) is a LangGraph requirement — the
graph diffs/merges state between node invocations.
"""

from typing import Literal, TypedDict

from app.models.schemas import Chunk, Citation, CriticScores

# Imported eagerly, not under TYPE_CHECKING: LangGraph resolves this
# TypedDict's annotations at runtime to build its state schema, so a string
# forward reference fails there. There is no import cycle to avoid —
# observability.tracing imports nothing from app.agents (the original
# `trace: object` annotation cited a cycle that does not exist, and cost the
# whole agent package its type checking).
from app.observability.tracing import TraceRecorder


class AgentState(TypedDict):
    query: str
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
    conversation_history: list[dict[str, str]]
