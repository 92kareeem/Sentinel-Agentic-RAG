"""A stand-in LLM for load tests: OpenAI chat-completions wire format, fixed latency.

Why it exists: a load test pointed at a hosted model measures the provider's
rate limiter (Groq's free tier allows ~30 requests a minute), not Sentinel.
This server answers like a fast hosted model, with configurable latency, so
the benchmark isolates what Sentinel itself costs: auth, guardrails, embedding,
access-filtered hybrid retrieval, the agent graph, the grounding gate, the
cache and the transport.

It answers well enough to pass Sentinel's real gates, which is the point —
a stub that produced refusals would benchmark the refusal path only:

  synthesizer  quotes the first prose sentences of the top retrieved passage,
               verbatim, each followed by its [chunk:id] tag, so the
               deterministic grounding gate verifies it for real
  critic       (the only json_mode caller) returns a passing verdict
  anything     else, e.g. the repair rewrite, echoes the user's message

Nothing about Sentinel is mocked: only the network hop to a model is replaced.

Run:  uvicorn mock_llm:app --port 9000
Env:  MOCK_LATENCY_MS  mean per-call latency (default 300, a typical hosted
                       small-model call)
      MOCK_JITTER      +/- fraction of that latency, uniform (default 0.2)
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time
from typing import Any

from fastapi import FastAPI, Request

LATENCY_S = float(os.environ.get("MOCK_LATENCY_MS", "300")) / 1000
JITTER = float(os.environ.get("MOCK_JITTER", "0.2"))

_EVIDENCE = re.compile(r"<<<BEGIN EVIDENCE>>>\n(.*?)\n<<<END EVIDENCE>>>", re.S)
_CHUNK = re.compile(r"^\[chunk:([\w-]+)\][^\n]*\n(.*?)(?=^\[chunk:|\Z)", re.S | re.M)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")

app = FastAPI(title="mock-llm", docs_url=None, redoc_url=None, openapi_url=None)


def _prose_sentences(text: str) -> list[str]:
    """Sentences of real prose: no headings, no table rows, at least six words."""
    lines = [ln.strip() for ln in text.splitlines()]
    prose = " ".join(ln for ln in lines if ln and not ln.startswith(("#", "|")))
    return [s.strip() for s in _SENTENCE.split(prose) if len(s.split()) >= 6]


def grounded_answer(prompt: str) -> str:
    """Two verbatim sentences from the best-ranked passage that has prose."""
    match = _EVIDENCE.search(prompt)
    if match:
        for chunk_id, text in _CHUNK.findall(match.group(1)):
            sentences = _prose_sentences(text)[:2]
            if sentences:
                return " ".join(f"{s} [chunk:{chunk_id}]" for s in sentences)
    return "The documents do not say."


def _reply(body: dict[str, Any]) -> str:
    messages = body.get("messages") or []
    last_user = next(
        (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), ""
    )
    if body.get("response_format", {}).get("type") == "json_object":
        return json.dumps({"faithfulness": 0.95, "relevance": 0.9, "why": "supported"})
    if "<<<BEGIN EVIDENCE>>>" in last_user:
        return grounded_answer(last_user)
    return str(last_user)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat(request: Request) -> dict[str, Any]:
    body = await request.json()
    await asyncio.sleep(LATENCY_S * random.uniform(1 - JITTER, 1 + JITTER))
    content = _reply(body)
    prompt_chars = sum(len(str(m.get("content", ""))) for m in body.get("messages") or [])
    return {
        "id": f"mock-{time.time_ns()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", "mock"),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        # ~4 characters per token, the usual English approximation.
        "usage": {
            "prompt_tokens": prompt_chars // 4,
            "completion_tokens": len(content) // 4,
            "total_tokens": (prompt_chars + len(content)) // 4,
        },
    }
