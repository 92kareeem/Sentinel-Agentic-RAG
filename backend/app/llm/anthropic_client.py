"""Anthropic Messages API adapter.

Role in architecture: the one provider whose wire format differs from the
chat-completions family. Differences handled here and nowhere else:

  * the system prompt is a top-level `system` field, not a message
  * `max_tokens` is required
  * the reply is a list of content blocks; text blocks are concatenated
  * usage is `input_tokens` / `output_tokens`
  * there is no JSON response mode, so json_mode becomes an instruction
    appended to the system prompt, and the parser strips any code fence
  * a `stop_reason` of "refusal" is a provider-side safety decline and is
    surfaced as a deterministic request error rather than retried

Spoken over the same resilient HTTP core as every other adapter rather than
through a vendor SDK, on purpose: the SDK runs its own retry loop with its
own timeouts, and the graph's deadline guarantee depends on exactly one
retry loop knowing how many seconds are left.
"""

from __future__ import annotations

import json
from typing import Any

from app.config import get_settings
from app.llm.base import LLMRequestError
from app.llm.http_chat import HttpRequest, post_chat

ANTHROPIC_VERSION = "2023-06-01"

_JSON_INSTRUCTION = (
    "\n\nRespond with a single JSON object and nothing else: no prose before or "
    "after it, no code fence."
)


def _split_system(messages: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
    system_parts = [m["content"] for m in messages if m.get("role") == "system"]
    turns = [m for m in messages if m.get("role") != "system"]
    return "\n\n".join(system_parts), turns


def _strip_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


def parse_messages_response(data: dict[str, Any]) -> tuple[str, int, int]:
    if data.get("stop_reason") == "refusal":
        raise LLMRequestError("Anthropic declined the request (stop_reason=refusal)")
    blocks = data.get("content") or []
    text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    usage = data.get("usage") or {}
    return text, int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))


def chat_completion(
    model: str,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    json_mode: bool = False,
    deadline_ts: float | None = None,
    max_retries: int | None = None,
) -> tuple[str, int, int]:
    settings = get_settings()
    if not model:
        raise ValueError("LLM_PROVIDER=anthropic requires ANTHROPIC_MODEL_SIMPLE/COMPLEX")
    system, turns = _split_system(messages)
    if json_mode:
        system = system + _JSON_INSTRUCTION
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": t["role"], "content": t["content"]} for t in turns],
    }
    if system:
        body["system"] = system

    def parse(data: dict[str, Any]) -> tuple[str, int, int]:
        text, tin, tout = parse_messages_response(data)
        if json_mode:
            text = _strip_fence(text)
            json.loads(text)  # a non-JSON body is a malformed response: let the core retry
        return text, tin, tout

    return post_chat(
        breaker_key=f"anthropic:{model}",
        request=HttpRequest(
            url=f"{settings.anthropic_base_url.rstrip('/')}/v1/messages",
            headers={
                "x-api-key": settings.anthropic_api_key,
                "anthropic-version": ANTHROPIC_VERSION,
                "Content-Type": "application/json",
            },
            body=body,
        ),
        parse=parse,
        deadline_ts=deadline_ts,
        max_retries=max_retries,
        provider_label="Anthropic",
    )
