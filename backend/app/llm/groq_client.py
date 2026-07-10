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


_breaker = CircuitBreaker()


def chat_completion(
    model: str,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    json_mode: bool = False,
) -> tuple[str, int, int]:
    """Call Groq chat completions. Returns (content, tokens_in, tokens_out).

    Retries on 429/5xx with exponential backoff + jitter, honoring
    Retry-After. Raises CircuitOpenError immediately if the breaker is open.
    """
    _breaker.before_call()
    settings = get_settings()
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
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
            if resp.status_code == 429 or resp.status_code >= 500:
                retry_after = float(resp.headers.get("Retry-After", 2**attempt))
                time.sleep(retry_after + random.uniform(0, 0.5))
                continue
            resp.raise_for_status()
            data = resp.json()
            _breaker.record_success()
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            return content, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        except (httpx.HTTPError, KeyError, IndexError) as exc:
            last_exc = exc
            time.sleep((2**attempt) + random.uniform(0, 0.5))

    _breaker.record_failure()
    raise RuntimeError(f"Groq call failed after {max_retries} retries") from last_exc
