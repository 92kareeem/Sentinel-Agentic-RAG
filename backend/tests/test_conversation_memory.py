import time
from unittest.mock import patch

from app.agents import synthesizer
from app.agents.state import AgentState
from app.observability.tracing import TraceRecorder


def test_synthesizer_includes_previous_conv_history_in_prompt() -> None:
    state: AgentState = {
        "query": "What about the refund window?",
        "user_id": "u1",
        "doc_id": None,
        "trace": TraceRecorder(),
        "attempt": 0,
        "model": "test-model",
        "token_budget_left": 1000,
        "deadline_ts": time.monotonic() + 10,
        "retrieved": [],
        "answer": "",
        "citations": [],
        "critic": None,
        "status": "running",
        "conversation_history": [
            {"role": "user", "content": "What is the refund policy?"},
            {"role": "assistant", "content": "Refunds take 3 days."},
        ],
    }

    def fake_chat(model: str, messages: list, **kwargs):
        prompt = messages[1]["content"]
        assert "PRIOR CONVERSATION" in prompt
        assert "What is the refund policy?" in prompt
        # history must be framed as context, never as evidence or instructions
        assert "never a source of facts" in prompt
        return "Follow-up answer", 5, 2

    with patch("app.llm.groq_client.chat_completion", side_effect=fake_chat):
        synthesizer.synthesizer_node(state)

    assert state["answer"] == "Follow-up answer"


def _state_with_history(history: list[dict]) -> AgentState:
    return {
        "query": "q",
        "user_id": "u1",
        "doc_id": None,
        "trace": TraceRecorder(),
        "attempt": 0,
        "model": "test-model",
        "token_budget_left": 1000,
        "deadline_ts": time.monotonic() + 10,
        "retrieved": [],
        "answer": "",
        "citations": [],
        "critic": None,
        "status": "running",
        "conversation_history": history,
    }


def test_history_is_bounded_in_turns_and_length() -> None:
    """Unbounded client-supplied history would crowd out retrieved evidence
    in the context window and inflate token cost on every request."""
    history = [{"role": "user", "content": f"turn-{i} " + "x" * 5000} for i in range(50)]
    prompt = synthesizer._build_user_prompt(_state_with_history(history))

    assert "turn-49" in prompt  # most recent kept
    assert "turn-0" not in prompt  # oldest dropped
    assert prompt.count("turn-") <= synthesizer._MAX_HISTORY_TURNS
    assert "x" * (synthesizer._MAX_HISTORY_CHARS + 50) not in prompt  # each turn truncated


def test_evidence_is_delimited_as_untrusted_data() -> None:
    """Document text must be fenced and labelled so injected 'instructions'
    inside an uploaded file read as quoted data, not as commands."""
    prompt = synthesizer._build_user_prompt(_state_with_history([]))
    assert "<<<BEGIN EVIDENCE>>>" in prompt and "<<<END EVIDENCE>>>" in prompt
    assert "untrusted" in prompt.lower()
