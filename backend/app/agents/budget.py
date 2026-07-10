"""Shared hard-cap check, called at the entry of every graph node.

Role in architecture: single source of truth for the three limits that turn
a runaway agent loop into a controlled refusal instead of a stuck Lambda:
attempt count, wall-clock deadline, and token budget.
"""

import time

from app.agents.state import AgentState
from app.config import get_settings


def check_budget(state: AgentState) -> bool:
    settings = get_settings()
    return (
        state["attempt"] <= settings.max_attempts
        and time.monotonic() < state["deadline_ts"]
        and state["token_budget_left"] > 0
    )
