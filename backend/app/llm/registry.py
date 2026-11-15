"""Provider dispatch: the one function the graph calls to talk to a model.

Role in architecture: agents/budget.llm_call -> registry.complete -> one
adapter. Adding a provider means adding one adapter module and one branch
here; nothing in agents/ or api/ changes.

The Groq branch calls `groq_client.chat_completion` through the module
attribute on purpose: the whole offline test suite stubs that one name, and
the eval harness drives it directly. Keeping it as the seam means a provider
abstraction did not cost a single existing test.
"""

from __future__ import annotations

import time
from typing import Any

from app.config import get_settings
from app.llm import anthropic_client, groq_client, openai_client, pricing
from app.llm.base import Completion


def provider_name() -> str:
    return get_settings().llm_provider


def complete(
    model: str,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    json_mode: bool = False,
    reasoning_effort: str = "low",
    deadline_ts: float | None = None,
    max_retries: int | None = None,
) -> Completion:
    """One chat completion against the configured provider, priced and timed."""
    provider = provider_name()
    common: dict[str, Any] = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "json_mode": json_mode,
        "deadline_ts": deadline_ts,
        "max_retries": max_retries,
    }
    t0 = time.perf_counter()
    if provider == "groq":
        content, tin, tout = groq_client.chat_completion(
            model=model, messages=messages, reasoning_effort=reasoning_effort, **common
        )
    elif provider == "anthropic":
        content, tin, tout = anthropic_client.chat_completion(
            model=model, messages=messages, **common
        )
    elif provider in ("openai", "azure", "local"):
        content, tin, tout = openai_client.chat_completion(
            model=model, messages=messages, flavour=provider, **common
        )
    else:  # pragma: no cover — Settings validates the literal
        raise ValueError(f"unknown LLM_PROVIDER {provider!r}")

    return Completion(
        content=content,
        tokens_in=tin,
        tokens_out=tout,
        provider=provider,
        model=model,
        cost_usd=pricing.estimate_cost(provider, model, tin, tout),
        latency_ms=int((time.perf_counter() - t0) * 1000),
    )
