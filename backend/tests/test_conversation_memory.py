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
        assert "Previous conversation" in messages[1]["content"]
        assert "What is the refund policy?" in messages[1]["content"]
        return "Follow-up answer", 5, 2

    with patch("app.llm.groq_client.chat_completion", side_effect=fake_chat):
        synthesizer.synthesizer_node(state)

    assert state["answer"] == "Follow-up answer"
