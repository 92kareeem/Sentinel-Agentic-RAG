"""LangGraph state schema — single source of truth for what flows between nodes.

Role in architecture: every node reads and returns this exact shape. Keeping
it a flat TypedDict (not nested Pydantic) is a LangGraph requirement — the
graph diffs/merges state between node invocations.
"""

from typing import Literal, TypedDict

from app.models.schemas import Chunk, Citation, CriticScores


class AgentState(TypedDict):
    query: str
    user_id: str
    trace: object  # observability.tracing.TraceRecorder; kept as object to avoid an import cycle
    attempt: int
    model: str
    token_budget_left: int
    deadline_ts: float
    retrieved: list[Chunk]
    answer: str
    citations: list[Citation]
    critic: CriticScores | None
    status: Literal["running", "answered", "refused"]
