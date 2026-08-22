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


class CircuitOpenError(RuntimeError):
    """Raised when the breaker is open; caller should refuse fast, not retry."""


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
    max_retries = 3
    last_exc: Exception | None = None

    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=15.0) as client:
                resp = client.post(GROQ_URL, json=body, headers=headers)
        except httpx.HTTPError as exc:  # connection/timeout — transient, retry
            last_exc = exc
            time.sleep((2**attempt) + random.uniform(0, 0.5))
            continue

        if resp.status_code == 429 or resp.status_code >= 500:  # transient, retry
            retry_after = float(resp.headers.get("Retry-After", 2**attempt))
            last_exc = RuntimeError(f"Groq returned {resp.status_code}")
            time.sleep(retry_after + random.uniform(0, 0.5))
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
            time.sleep((2**attempt) + random.uniform(0, 0.5))
            continue

        breaker.record_success()
        return content, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)

    breaker.record_failure()
    raise RuntimeError(f"Groq call failed after {max_retries} retries") from last_exc
