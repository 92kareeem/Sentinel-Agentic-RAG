"""Shared hard-cap check, called at the entry of every graph node.

Role in architecture: single source of truth for the three limits that turn
a runaway agent loop into a controlled refusal instead of a stuck Lambda:
attempt count, wall-clock deadline, and token budget.
"""

import time
from typing import Any

from app.agents.state import AgentState
from app.config import get_settings


def check_budget(state: AgentState) -> bool:
    settings = get_settings()
    return (
        state["attempt"] <= settings.max_attempts
        and time.monotonic() < state["deadline_ts"]
        and state["token_budget_left"] > 0
    )


def llm_call(state: AgentState, **kwargs: Any) -> tuple[str, int, int] | None:
    """An LLM call bounded by what remains of THIS request's wall clock.

    Returns None when the deadline ran out mid-call, having already marked the
    state refused — callers return immediately and the graph routes to
    refusal_node, which reports BUDGET_EXHAUSTED.

    Why this exists at all: check_budget() above runs BETWEEN nodes, so a node
    already inside a provider call could not be interrupted. Groq answers a
    free-tier token-per-minute limit with a Retry-After measured in tens of
    seconds, and the client retried three times per call across several nodes.
    A 45-second budget was measured taking 285 seconds. In Lambda that is not a
    slow answer — it is a function timeout and a 502 with no refusal at all.

    Every LLM call in the graph goes through here so the bound cannot be
    forgotten at one call site, which is the same reason groq_client exists
    rather than three copies of the retry logic.
    """
    from app.llm import groq_client

    try:
        return groq_client.chat_completion(deadline_ts=state["deadline_ts"], **kwargs)
    except groq_client.GroqDeadlineError:
        state["status"] = "refused"
        return None
