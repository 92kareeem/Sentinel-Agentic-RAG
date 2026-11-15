"""The one resilient HTTP loop every chat adapter runs through.

Role in architecture: timeout, retry with jittered backoff, Retry-After
handling, deadline bounding and the per-model circuit breaker exist exactly
once, here. An adapter supplies three things — how to build the request, how
to read the response, which breaker to use — and inherits all of the above.

Why the deadline is enforced INSIDE the call: agents/budget.check_budget runs
between graph nodes, so a node already inside a provider call could not be
interrupted. A free-tier provider answers a token-per-minute limit with a
Retry-After measured in tens of seconds, and three retries across several
nodes turned a measured 45-second budget into a 285-second request. In Lambda
that is a function timeout and a 502 with no refusal at all.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

from app.config import get_settings
from app.llm.base import LLMDeadlineError, LLMRequestError, breaker_for

# Ceiling on any single HTTP attempt, independent of the caller's deadline. A
# request that has not answered in this long is not about to.
MAX_CALL_SECONDS = 15.0


@dataclass(frozen=True)
class HttpRequest:
    url: str
    headers: dict[str, str]
    body: dict[str, Any]


Parser = Callable[[dict[str, Any]], tuple[str, int, int]]


def _remaining(deadline_ts: float | None) -> float | None:
    """Seconds left before the caller's deadline, or None if unbounded."""
    if deadline_ts is None:
        return None
    return deadline_ts - time.monotonic()


def post_chat(
    *,
    breaker_key: str,
    request: HttpRequest,
    parse: Parser,
    deadline_ts: float | None,
    max_retries: int | None,
    provider_label: str,
) -> tuple[str, int, int]:
    """POST once per attempt until a usable body comes back, or give up.

    Retries on 429/5xx and connection errors with exponential backoff and
    jitter, honouring Retry-After when the caller can still afford to wait.
    A deterministic 4xx raises LLMRequestError at once. A deadline that would
    be crossed by the next sleep raises LLMDeadlineError instead of sleeping
    into it: the answer is discarded either way, and refusing early leaves
    time to say so.
    """
    breaker = breaker_for(breaker_key)
    breaker.before_call()
    if max_retries is None:
        max_retries = get_settings().llm_max_retries
    last_exc: Exception | None = None

    def _sleep_or_give_up(seconds: float) -> None:
        left = _remaining(deadline_ts)
        if left is not None and seconds >= left:
            raise LLMDeadlineError(
                f"no time left to retry: needs {seconds:.1f}s, {left:.1f}s remain"
            )
        time.sleep(seconds)

    for attempt in range(max_retries):
        left = _remaining(deadline_ts)
        if left is not None and left <= 0:
            breaker.record_failure()
            raise LLMDeadlineError("deadline passed before the call was attempted")

        # Never wait longer than the caller can afford, and never longer than
        # the fixed ceiling — a single hung connection must not consume a
        # budget that three nodes still have to share.
        timeout = MAX_CALL_SECONDS if left is None else min(MAX_CALL_SECONDS, left)
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(request.url, json=request.body, headers=request.headers)
        except httpx.HTTPError as exc:  # connection/timeout — transient, retry
            last_exc = exc
            _sleep_or_give_up((2**attempt) + random.uniform(0, 0.5))
            continue

        if resp.status_code == 429 or resp.status_code >= 500:  # transient, retry
            # The provider's Retry-After is authoritative about when IT will
            # serve us, and unbounded. Honour it, but only if we can afford to.
            retry_after = float(resp.headers.get("Retry-After", 2**attempt))
            last_exc = RuntimeError(f"{provider_label} returned {resp.status_code}")
            _sleep_or_give_up(retry_after + random.uniform(0, 0.5))
            continue

        if resp.status_code >= 400:  # deterministic — retrying can't help
            breaker.record_failure()
            raise LLMRequestError(
                f"{provider_label} rejected request ({resp.status_code}): {resp.text[:300]}"
            )

        try:
            parsed = parse(resp.json())
        except (ValueError, KeyError, IndexError, TypeError) as exc:  # malformed 200 — rare
            last_exc = exc
            _sleep_or_give_up((2**attempt) + random.uniform(0, 0.5))
            continue

        breaker.record_success()
        return parsed

    breaker.record_failure()
    raise RuntimeError(f"{provider_label} call failed after {max_retries} retries") from last_exc


def parse_openai_style(data: dict[str, Any]) -> tuple[str, int, int]:
    """The chat-completions response shape shared by Groq, OpenAI, Azure,
    Ollama and vLLM: choices[0].message.content plus usage.prompt_tokens /
    usage.completion_tokens."""
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage") or {}
    return str(content or ""), int(usage.get("prompt_tokens", 0)), int(
        usage.get("completion_tokens", 0)
    )
