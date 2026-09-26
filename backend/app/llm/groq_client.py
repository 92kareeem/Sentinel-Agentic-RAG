"""Groq LLM client: timeout, retry with backoff, and a circuit breaker.

Role in architecture: the only module that speaks to Groq's API. Every LLM
call in the agent graph goes through here so retry/backoff/circuit-breaker
logic exists exactly once, not copy-pasted into router/synthesizer/critic.
"""

import random
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.config import get_settings

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Ceiling on any single HTTP attempt, independent of the caller's deadline. A
# request that has not answered in this long is not about to.
MAX_CALL_SECONDS = 15.0


class CircuitOpenError(RuntimeError):
    """Raised when the breaker is open; caller should refuse fast, not retry."""


class GroqDeadlineError(RuntimeError):
    """The request's wall-clock budget ran out inside this call.

    Distinct from a transport failure: nothing is wrong with Groq or with the
    request, there is simply no time left to finish it. Callers turn this into
    a BUDGET_EXHAUSTED refusal, which is retryable and says nothing about
    whether the documents cover the question.
    """


def _remaining(deadline_ts: float | None) -> float | None:
    """Seconds left before the caller's deadline, or None if unbounded."""
    if deadline_ts is None:
        return None
    return deadline_ts - time.monotonic()


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


class GroqRequestError(RuntimeError):
    """A deterministic client-side rejection (4xx other than 429).

    Retrying an identical request against the same error (bad model name,
    malformed body, auth failure) can't succeed — it only burns the repair
    loop's deadline and token budget on calls that were never going to work.
    Raised immediately, without retry or backoff.
    """


# One breaker per model: a struggling 20b shouldn't trip the breaker for the
# 120b escalation path (or vice versa) — they're independent upstream
# resources, and repair_escalate exists specifically to route around a
# problem with the first model.
_breakers: dict[str, CircuitBreaker] = {}


def _get_breaker(model: str) -> CircuitBreaker:
    if model not in _breakers:
        _breakers[model] = CircuitBreaker()
    return _breakers[model]


def chat_completion(
    model: str,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    json_mode: bool = False,
    reasoning_effort: str = "low",
    deadline_ts: float | None = None,
    max_retries: int | None = None,
) -> tuple[str, int, int]:
    """Call Groq chat completions. Returns (content, tokens_in, tokens_out).

    Retries on 429/5xx (transient) and connection errors, with exponential
    backoff + jitter honoring Retry-After. A deterministic 4xx (bad request,
    auth, unknown model) raises GroqRequestError immediately — retrying it
    cannot succeed. Raises CircuitOpenError immediately if this model's
    breaker is open.

    reasoning_effort caps the hidden chain-of-thought tokens that Groq's
    open-weight models (openai/gpt-oss-*) emit before the visible content —
    without this, short max_tokens budgets (router/critic) get exhausted by
    reasoning alone and return truncated or empty content.

    max_retries is low by default because an interactive request has a person
    waiting: more attempts only spend their patience. A batch caller that can
    genuinely afford to wait out a rate limit (the eval harness) raises it,
    and pairs it with a deadline so "wait" still has a bound.

    deadline_ts (time.monotonic) bounds the WHOLE call, retries and backoff
    sleeps included. Without it the graph's deadline was advisory: check_budget
    runs between nodes, so a node already inside this function could not be
    interrupted, and Groq's free tier answers a token-per-minute limit with a
    Retry-After measured in tens of seconds. Three retries here, across several
    nodes, turned a 45-second budget into a measured 285-second request — and
    in Lambda that is not a slow answer, it is a function timeout and a 502
    with no refusal at all.
    """
    breaker = _get_breaker(model)
    breaker.before_call()
    settings = get_settings()
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "reasoning_effort": reasoning_effort,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}

    headers = {
        "Authorization": f"Bearer {settings.groq_api_key}",
        "Content-Type": "application/json",
    }
    if max_retries is None:
        max_retries = settings.llm_max_retries
    last_exc: Exception | None = None

    def _sleep_or_give_up(seconds: float) -> None:
        """Back off, unless doing so would outlast the caller's deadline.

        Sleeping past the deadline is strictly worse than failing now: the
        answer is discarded either way, and the caller spends the difference
        holding a connection open. Refusing early leaves time to say so.
        """
        left = _remaining(deadline_ts)
        if left is not None and seconds >= left:
            raise GroqDeadlineError(
                f"no time left to retry: needs {seconds:.1f}s, {left:.1f}s remain"
            )
        time.sleep(seconds)

    for attempt in range(max_retries):
        left = _remaining(deadline_ts)
        if left is not None and left <= 0:
            breaker.record_failure()
            raise GroqDeadlineError("deadline passed before the call was attempted")

        # Never wait longer than the caller can afford, and never longer than
        # the fixed ceiling — a single hung connection must not consume a
        # budget that three nodes still have to share.
        timeout = MAX_CALL_SECONDS if left is None else min(MAX_CALL_SECONDS, left)
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(GROQ_URL, json=body, headers=headers)
        except httpx.HTTPError as exc:  # connection/timeout — transient, retry
            last_exc = exc
            _sleep_or_give_up((2**attempt) + random.uniform(0, 0.5))
            continue

        if resp.status_code == 429 or resp.status_code >= 500:  # transient, retry
            # Groq's Retry-After is authoritative about when IT will serve us,
            # and unbounded. Honour it, but only if we can afford to wait.
            retry_after = float(resp.headers.get("Retry-After", 2**attempt))
            last_exc = RuntimeError(f"Groq returned {resp.status_code}")
            _sleep_or_give_up(retry_after + random.uniform(0, 0.5))
            continue

        if resp.status_code >= 400:  # deterministic — retrying can't help
            breaker.record_failure()
            raise GroqRequestError(f"Groq rejected request ({resp.status_code}): {resp.text[:300]}")

        try:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
        except (ValueError, KeyError, IndexError) as exc:  # malformed 200 body — rare, retry
            last_exc = exc
            _sleep_or_give_up((2**attempt) + random.uniform(0, 0.5))
            continue

        breaker.record_success()
        return content, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)

    breaker.record_failure()
    raise RuntimeError(f"Groq call failed after {max_retries} retries") from last_exc
