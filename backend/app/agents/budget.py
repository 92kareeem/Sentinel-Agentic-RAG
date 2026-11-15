"""Shared hard-cap check, called at the entry of every graph node.

Role in architecture: single source of truth for the three limits that turn
a runaway agent loop into a controlled refusal instead of a stuck Lambda:
attempt count, wall-clock deadline, and token budget. Also the one place a
model call is priced, and the one place the model can be downgraded for cost.
"""

from __future__ import annotations

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


def seconds_left(state: AgentState) -> float:
    return max(0.0, state["deadline_ts"] - time.monotonic())


def pick_model(state: AgentState) -> tuple[str, bool]:
    """The model this node should actually call, and whether it was downgraded.

    The router (ADR 0002) and repair_escalate choose state["model"] on quality
    grounds. This applies a cost rule on top: when the remaining wall clock or
    token budget is thin, the cheap model is used regardless, because a large
    model that cannot finish inside the deadline produces a BUDGET_EXHAUSTED
    refusal, which is strictly worse than a small model's verified answer.

    Deterministic by design, like the router. A budget decision made by an LLM
    would spend tokens on exactly the requests that have none to spare.
    """
    settings = get_settings()
    chosen = state["model"]
    cheap = settings.model_simple
    if chosen == cheap:
        return chosen, False
    tight = (
        seconds_left(state) < settings.downgrade_deadline_seconds
        or state["token_budget_left"] < settings.downgrade_token_threshold
    )
    return (cheap, True) if tight else (chosen, False)


def llm_call(state: AgentState, **kwargs: Any) -> tuple[str, int, int] | None:
    """An LLM call bounded by what remains of THIS request's wall clock.

    Returns None when the deadline ran out mid-call, having already marked the
    state refused — callers return immediately and the graph routes to
    refusal_node, which reports BUDGET_EXHAUSTED.

    Every LLM call in the graph goes through here so the bound cannot be
    forgotten at one call site, the estimated cost is accumulated on the trace
    once, and provider selection lives in one module (llm/registry.py).
    """
    from app.llm import registry
    from app.llm.base import LLMDeadlineError

    try:
        completion = registry.complete(deadline_ts=state["deadline_ts"], **kwargs)
    except LLMDeadlineError:
        state["status"] = "refused"
        return None
    state["trace"].add_cost(completion.cost_usd, completion.provider)
    return completion.as_tuple
