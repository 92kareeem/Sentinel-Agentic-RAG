"""The cost rule that sits on top of model selection.

The router and escalation pick a model on quality grounds. When the request
is nearly out of time or tokens, a large model that cannot finish produces
a BUDGET_EXHAUSTED refusal, which is worse than a cheap model's verified
answer. pick_model() makes that trade deterministically and records it.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import patch

from app.agents import retriever
from app.agents.budget import pick_model
from app.agents.graph import build_graph
from app.config import get_settings
from app.models.schemas import Chunk
from app.observability.tracing import TraceRecorder


def _state(**over: Any) -> Any:
    s = get_settings()
    base: dict[str, Any] = {
        "query": "q", "search_query": "q", "user_id": "u", "doc_id": None,
        "trace": TraceRecorder(), "attempt": 0, "model": s.model_complex,
        "token_budget_left": s.token_budget, "deadline_ts": time.monotonic() + 60,
        "retrieved": [], "answer": "", "citations": [], "critic": None,
        "status": "running", "refusal_reason": None, "conversation_history": [],
    }
    base.update(over)
    return base


def test_a_comfortable_budget_keeps_the_chosen_model() -> None:
    model, downgraded = pick_model(_state())
    assert model == get_settings().model_complex and downgraded is False


def test_a_thin_deadline_downgrades_to_the_cheap_model() -> None:
    model, downgraded = pick_model(_state(deadline_ts=time.monotonic() + 2))
    assert model == get_settings().model_simple and downgraded is True


def test_a_thin_token_budget_downgrades_to_the_cheap_model() -> None:
    model, downgraded = pick_model(_state(token_budget_left=100))
    assert model == get_settings().model_simple and downgraded is True


def test_the_cheap_model_is_never_reported_as_downgraded() -> None:
    model, downgraded = pick_model(
        _state(model=get_settings().model_simple, token_budget_left=1)
    )
    assert downgraded is False and model == get_settings().model_simple


def test_the_trace_records_the_model_actually_called_and_the_cost() -> None:
    """A trace that reports the model the router chose, rather than the one
    that ran, is how a broken component looks healthy (ADR 0002)."""
    chunk = Chunk(chunk_id="c1", doc_id="d", section_path="S", text="Refunds take 3 days.",
                  token_count=5, char_start=0, char_end=20)

    def fake_retriever(state: Any) -> Any:
        state["retrieved"] = [chunk]
        state["trace"].record_step("retriever", 0)
        return state

    called: list[str] = []

    def fake_chat(model: str, messages: list, **kw: Any) -> tuple[str, int, int]:
        called.append(model)
        if "grading judge" in messages[0]["content"]:
            return '{"faithfulness": 0.9, "relevance": 0.9, "why": "ok"}', 50, 20
        return "Refunds take 3 days [chunk:c1].", 100, 30

    # "compare" routes to the large model (router heuristic); the thin deadline
    # then forces the downgrade this test is about.
    state = _state(
        query="compare refund windows", search_query="compare refund windows",
        deadline_ts=time.monotonic() + 3,  # under downgrade_deadline_seconds
    )
    with (
        patch.object(retriever, "retriever_node", fake_retriever),
        patch("app.llm.groq_client.chat_completion", fake_chat),
    ):
        result = build_graph().invoke(state)

    assert result["status"] == "answered"
    assert called[0] == get_settings().model_simple  # synthesizer was downgraded
    synth = next(s for s in state["trace"].steps if s["name"] == "synthesizer")
    assert synth["meta"]["model"] == get_settings().model_simple
    assert synth["meta"]["downgraded"] == "True"
    assert state["trace"].est_cost_usd > 0
    assert state["trace"].to_dict("answered")["provider"] == "groq"
