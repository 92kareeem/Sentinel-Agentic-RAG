"""P2 verification: force the repair loop to fire.

A healthy tiny corpus rarely fails organically, so this script patches the
critic to reject attempt 0 (as if the first answer scored poorly). The graph
must then route critic -> repair_rewrite -> retriever -> synthesizer ->
critic, and pass on attempt 1. Everything except the critic's first verdict
is real: real retrieval, real Groq calls, real rewrite.
"""

import time
from unittest.mock import patch

from app.agents import critic as critic_mod
from app.agents.graph import build_graph
from app.agents.state import AgentState
from app.config import get_settings
from app.models.schemas import CriticScores
from app.observability.tracing import TraceRecorder

_real_critic = critic_mod.critic_node


def failing_first_critic(state: AgentState) -> AgentState:
    if state["attempt"] == 0:
        scores = CriticScores(faithfulness=0.2, relevance=0.3)
        state["critic"] = scores
        state["trace"].critic_scores.append({**scores.model_dump(), "attempt": 0})
        state["trace"].record_step("critic", 0, why="forced failure to demo repair loop")
        return state
    return _real_critic(state)


def main() -> None:
    settings = get_settings()
    query = "What are the refund conditions?"
    trace = TraceRecorder(query_redacted=query)
    state: AgentState = {
        "query": query,
        "user_id": "local",
        "doc_id": None,
        "trace": trace,
        "attempt": 0,
        "model": settings.groq_model_simple,
        "token_budget_left": settings.token_budget,
        "deadline_ts": time.monotonic() + settings.deadline_seconds,
        "retrieved": [],
        "answer": "",
        "citations": [],
        "critic": None,
        "status": "running",
    }
    with patch.object(critic_mod, "critic_node", failing_first_critic):
        result = build_graph().invoke(state)

    print(f"Q: {query}")
    print(f"status: {result['status']} | repairs: {trace.repair_count}")
    print(f"critic history: {trace.critic_scores}")
    print(f"rewritten query: {result['query']}")
    print(f"answer: {result['answer']}")
    print("node sequence:", " -> ".join(s["name"] for s in trace.steps))
    assert trace.repair_count >= 1, "repair loop did not fire!"
    assert result["status"] in ("answered", "refused")
    print("\nREPAIR LOOP VERIFIED")


if __name__ == "__main__":
    main()
