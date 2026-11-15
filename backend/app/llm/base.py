"""Shared vocabulary for every LLM adapter.

Role in architecture: the graph never speaks to a provider directly. It calls
agents/budget.llm_call, which calls llm/registry.complete, which picks one of
the adapters in this package. Every adapter returns the same Completion and
raises the same three errors, so the graph's behaviour under failure — refuse
on deadline, fail fast on a bad request, trip a breaker on repeated outages —
is identical whichever vendor is behind it.

The errors keep their historical Groq-era aliases in groq_client.py so the
existing tests and the eval harness keep importing the names they know.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


class CircuitOpenError(RuntimeError):
    """Raised when the breaker is open; caller should refuse fast, not retry."""


class LLMDeadlineError(RuntimeError):
    """The request's wall-clock budget ran out inside this call.

    Distinct from a transport failure: nothing is wrong with the provider or
    the request, there is simply no time left to finish it. Callers turn this
    into a BUDGET_EXHAUSTED refusal, which is retryable and says nothing about
    whether the documents cover the question.
    """


class LLMRequestError(RuntimeError):
    """A deterministic client-side rejection (4xx other than 429).

    Retrying an identical request against the same error (bad model name,
    malformed body, auth failure) can't succeed — it only burns the repair
    loop's deadline and token budget on calls that were never going to work.
    Raised immediately, without retry or backoff.
    """


@dataclass(frozen=True)
class Completion:
    """One provider response, normalised.

    cost_usd is an ESTIMATE from the pricing table, not a bill. It exists so
    the trace, the telemetry tab and the budget downgrade rule all read one
    number computed one way.
    """

    content: str
    tokens_in: int
    tokens_out: int
    provider: str
    model: str
    cost_usd: float = 0.0
    latency_ms: int = 0

    @property
    def as_tuple(self) -> tuple[str, int, int]:
        """The shape every graph node consumes."""
        return self.content, self.tokens_in, self.tokens_out


@dataclass
class CircuitBreaker:
    fail_threshold: int = 3
    open_seconds: float = 30.0
    _failures: int = field(default=0, init=False)
    _opened_at: float | None = field(default=None, init=False)

    def before_call(self) -> None:
        if self._opened_at is not None:
            if time.monotonic() - self._opened_at < self.open_seconds:
                raise CircuitOpenError("circuit open; failing fast")
            self._opened_at = None  # half-open: allow one probe
            self._failures = 0

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.fail_threshold:
            self._opened_at = time.monotonic()


# One breaker per (provider, model): a struggling small model must not trip
# the breaker for the escalation model, which is exactly the path
# repair_escalate exists to route around a problem with the first one.
BREAKERS: dict[str, CircuitBreaker] = {}


def breaker_for(key: str) -> CircuitBreaker:
    if key not in BREAKERS:
        BREAKERS[key] = CircuitBreaker()
    return BREAKERS[key]
