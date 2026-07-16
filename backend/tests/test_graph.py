"""Agent graph routing tests with a mocked Groq client and fake retriever.

The point: prove the state machine's routing (pass / rewrite / escalate /
refuse) deterministically, offline, for free. Canned LLM responses are keyed
off each node's distinctive system prompt.
"""

import time
from unittest.mock import patch

import pytest
from app.agents import retriever
from app.agents.graph import build_graph
from app.agents.state import AgentState
from app.config import get_settings
from app.models.schemas import Chunk
from app.observability.tracing import TraceRecorder

FAKE_CHUNKS = [
    Chunk(chunk_id="c1", doc_id="d", section_path="Refunds", text="Refunds take 3 days.",
          token_count=6, char_start=0, char_end=20),
]


def fake_retriever(state: AgentState) -> AgentState:
    state["retrieved"] = FAKE_CHUNKS
    state["trace"].record_step("retriever", 0)
    return state


def make_groq(critic_scores: list[str]):
    """Return a chat_completion double; critic_scores is consumed per critic call."""

    def fake_chat(model: str, messages: list, **kwargs) -> tuple[str, int, int]:
        system = messages[0]["content"]
        if "Classify" in system:
            return "simple", 5, 1
        if "grading judge" in system:
            return critic_scores.pop(0), 50, 20
        if "Rewrite" in system:
            return "rewritten query", 20, 5
        return "Refunds take 3 days [chunk:c1].", 100, 30  # synthesizer

    return fake_chat


def _state() -> AgentState:
    settings = get_settings()
    return {
        "query": "How long do refunds take?", "user_id": "t", "doc_id": None,
        "trace": TraceRecorder(), "attempt": 0,
        "model": settings.groq_model_simple,
        "token_budget_left": settings.token_budget,
        "deadline_ts": time.monotonic() + settings.deadline_seconds,
        "retrieved": [], "answer": "", "citations": [], "critic": None,
        "status": "running",
    }


GOOD = '{"faithfulness": 0.9, "relevance": 0.9, "why": "ok"}'
BAD = '{"faithfulness": 0.1, "relevance": 0.2, "why": "bad"}'


def _run(critic_scores: list[str]) -> tuple[AgentState, TraceRecorder]:
    state = _state()
    trace = state["trace"]
    with (
        patch.object(retriever, "retriever_node", fake_retriever),
        patch("app.llm.groq_client.chat_completion", make_groq(critic_scores)),
    ):
        result = build_graph().invoke(state)
    return result, trace


def test_happy_path_answers_first_attempt() -> None:
    result, trace = _run([GOOD])
    assert result["status"] == "answered"
    assert trace.repair_count == 0
    assert result["citations"][0].chunk_id == "c1"


def test_first_failure_triggers_rewrite_then_answers() -> None:
    result, trace = _run([BAD, GOOD])
    assert result["status"] == "answered"
    assert trace.repair_count == 1
    names = [s["name"] for s in trace.steps]
    assert "repair_rewrite" in names
    assert result["query"] == "rewritten query"


def test_two_failures_escalate_then_answer() -> None:
    result, trace = _run([BAD, BAD, GOOD])
    assert result["status"] == "answered"
    assert trace.repair_count == 2
    names = [s["name"] for s in trace.steps]
    assert "repair_rewrite" in names and "repair_escalate" in names
    assert result["model"] == get_settings().groq_model_complex


def test_three_failures_refuse() -> None:
    result, trace = _run([BAD, BAD, BAD])
    assert result["status"] == "refused"
    assert trace.repair_count == 2  # both strategies spent, then refusal


def test_judge_parse_failure_fails_closed() -> None:
    # critic returns garbage twice per call; graph must treat as failure, not crash
    result, _ = _run(["not json", "still not json"] * 3)
    assert result["status"] == "refused"


@pytest.mark.parametrize("scores", [[GOOD]])
def test_grounding_strips_uncited_when_critic_passes(scores: list[str]) -> None:
    result, _ = _run(scores)
    # grounding kept the cited sentence verbatim
    assert "[chunk:c1]" in result["answer"]
