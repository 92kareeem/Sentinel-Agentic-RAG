"""Groq adapter (OpenAI-compatible chat completions, ultra-low latency).

Role in architecture: the original and default provider. The resilience
logic that used to live here is now llm/http_chat.py, shared by every
adapter; this module only knows Groq's URL, auth header and the one
Groq-specific knob (reasoning_effort for the gpt-oss models).

The Groq-era error names are kept as aliases so the eval harness, the tests
and any operator script that imports them keep working.
"""

from __future__ import annotations

import time  # noqa: F401 — tests patch groq_client.time.sleep
from typing import Any

import httpx  # noqa: F401 — tests patch groq_client.httpx.Client

from app.config import get_settings
from app.llm import base
from app.llm.base import CircuitOpenError, LLMDeadlineError, LLMRequestError
from app.llm.http_chat import (
    MAX_CALL_SECONDS,  # noqa: F401 — re-exported; the ceiling is read from here by callers
    HttpRequest,
    parse_openai_style,
    post_chat,
)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
PROVIDER = "groq"

# Backwards-compatible names. New code should use the llm.base names.
GroqDeadlineError = LLMDeadlineError
GroqRequestError = LLMRequestError
__all__ = [
    "CircuitOpenError",
    "GroqDeadlineError",
    "GroqRequestError",
    "chat_completion",
]

# The same dict object as llm.base.BREAKERS: tests clear it between cases.
_breakers = base.BREAKERS


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

    reasoning_effort caps the hidden chain-of-thought tokens that Groq's
    open-weight models (openai/gpt-oss-*) emit before the visible content —
    without this, short max_tokens budgets (critic) get exhausted by
    reasoning alone and return truncated or empty content.
    """
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
    request = HttpRequest(
        url=GROQ_URL,
        headers={
            "Authorization": f"Bearer {settings.groq_api_key}",
            "Content-Type": "application/json",
        },
        body=body,
    )
    return post_chat(
        breaker_key=model,  # historical key shape: tests assert per-model isolation
        request=request,
        parse=parse_openai_style,
        deadline_ts=deadline_ts,
        max_retries=max_retries,
        provider_label="Groq",
    )
