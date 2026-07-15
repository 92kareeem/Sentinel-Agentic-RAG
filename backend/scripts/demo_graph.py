"""P2 verification script: run 3 scripted questions through the full agent graph.

Not part of the production API — proves router -> retriever -> synthesizer ->
critic -> repair works end-to-end against the P1 index and a real Groq key.
Question 3 is deliberately vague so weak retrieval forces a repair attempt.
"""

import time

from app.agents.graph import build_graph
from app.agents.state import AgentState
from app.config import get_settings
from app.observability.tracing import TraceRecorder

QUESTIONS = [
    "What are the refund conditions for defective items?",
    "What laptop budget does an Engineer get and how often does it refresh?",
    "What about the thing with the stuff for the process?",  # vague -> should trigger repair
]


def run(query: str) -> None:
    settings = get_settings()
    graph = build_graph()
    trace = TraceRecorder(query_redacted=query)
    state: AgentState = {
        "query": query,
        "user_id": "local",
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
    result = graph.invoke(state)

    print(f"\n{'=' * 70}\nQ: {query}")
    print(
        f"status: {result['status']} | repairs: {trace.repair_count} "
        f"| model_path: {trace.model_path}"
    )
    if result["critic"]:
        print(
            f"critic: faithfulness={result['critic'].faithfulness:.2f} "
            f"relevance={result['critic'].relevance:.2f}"
        )
    print(f"answer: {result['answer']}")
    print(f"citations: {[c.chunk_id for c in result['citations']]}")


if __name__ == "__main__":
    for q in QUESTIONS:
        run(q)
