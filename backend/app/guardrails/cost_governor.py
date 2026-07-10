"""Cost governor: stamps the per-request token budget and deadline.

Role in architecture: the last guardrail before the graph runs. It doesn't
reject anything itself — it hands the graph the hard caps that
agents/budget.check_budget enforces at every node entry.
"""

import time
from dataclasses import dataclass

from app.config import get_settings


@dataclass(frozen=True)
class RequestBudget:
    token_budget: int
    deadline_ts: float  # time.monotonic() based


def allocate_budget() -> RequestBudget:
    settings = get_settings()
    return RequestBudget(
        token_budget=settings.token_budget,
        deadline_ts=time.monotonic() + settings.deadline_seconds,
    )
