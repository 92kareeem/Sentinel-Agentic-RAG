"""The judge's reply is validated as a verdict, not merely parsed as JSON.

Three malformed shapes were reproduced from a real judge — a list, a score
of "high", a score of 7 — and each used to raise past the parse-failure
handling and turn an already-drafted answer into a 503. They now cost one
re-ask, then fail closed, and their tokens are always deducted.
"""

from __future__ import annotations

import time
from typing import Any

import pytest
from app.agents.critic import InvalidVerdict, critic_node, parse_verdict
from app.config import get_settings
from app.llm import groq_client
from app.models.schemas import Chunk
from app.observability.tracing import TraceRecorder


@pytest.mark.parametrize(
    "bad",
    [
        "not json",
        "[0.9, 0.9]",
        '{"faithfulness": "high", "relevance": 0.9}',
        '{"faithfulness": 7, "relevance": 0.9}',
        '{"faithfulness": 0.9}',
        '{"faithfulness": -0.1, "relevance": 0.5}',
    ],
)
def test_malformed_verdicts_are_rejected(bad: str) -> None:
    with pytest.raises(InvalidVerdict):
        parse_verdict(bad)


def test_a_numeric_string_is_tolerated() -> None:
    v = parse_verdict('{"faithfulness": "0.8", "relevance": 1, "why": "fine"}')
    assert (v.faithfulness, v.relevance) == (0.8, 1.0)


def _state() -> Any:
    s = get_settings()
    return {
        "query": "q", "search_query": "q", "user_id": "u", "doc_id": None,
        "trace": TraceRecorder(), "attempt": 0, "model": s.model_simple,
        "token_budget_left": 1000, "deadline_ts": time.monotonic() + 60,
        "retrieved": [Chunk(chunk_id="c", doc_id="d", section_path="S", text="t",
                            token_count=1, char_start=0, char_end=1)],
        "answer": "An answer. [chunk:c]", "citations": [], "critic": None,
        "status": "running", "refusal_reason": None, "conversation_history": [],
    }


def test_two_invalid_verdicts_fail_closed_and_are_paid_for(monkeypatch) -> None:
    replies = iter(['{"faithfulness": 7, "relevance": 1}', "[1, 1]"])
    monkeypatch.setattr(groq_client, "chat_completion", lambda **_: (next(replies), 40, 10))
    state = _state()
    out = critic_node(state)
    assert out["critic"].faithfulness == 0.0 and out["critic"].relevance == 0.0
    # Both calls deducted: 2 x (40 + 10). This used to be 0 because the
    # parse raised before usage was returned.
    assert out["token_budget_left"] == 1000 - 100
    step = next(s for s in state["trace"].steps if s["name"] == "critic")
    assert step["tokens_in"] == 80 and "invalid verdict" in step["meta"]["why"]


def test_one_invalid_verdict_then_a_good_one_passes(monkeypatch) -> None:
    replies = iter(["garbage", '{"faithfulness": 0.95, "relevance": 0.9, "why": "ok"}'])
    monkeypatch.setattr(groq_client, "chat_completion", lambda **_: (next(replies), 10, 5))
    out = critic_node(_state())
    assert out["critic"].faithfulness == 0.95
    assert out["token_budget_left"] == 1000 - 30
