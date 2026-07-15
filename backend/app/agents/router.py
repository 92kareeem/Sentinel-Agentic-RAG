"""Router node: classifies query complexity, picks a Groq model.

Role in architecture: one cheap LLM call decides whether the query needs the
8b model (fast/cheap) or the 70b model (multi-hop, tables). Falls back to a
keyword heuristic if the classifier call fails — the graph degrades
gracefully instead of crashing when Groq has a bad moment.
"""

import time

from app.agents.budget import check_budget
from app.agents.state import AgentState
from app.config import get_settings
from app.llm import groq_client

_HEURISTIC_COMPLEX = ("compare", "difference between", "table", "how many", " vs ")


def _heuristic(query: str) -> str:
    q = query.lower()
    return "multi_hop" if any(k in q for k in _HEURISTIC_COMPLEX) else "simple"


def router_node(state: AgentState) -> AgentState:
    if not check_budget(state):
        state["status"] = "refused"
        return state

    settings = get_settings()
    t0 = time.perf_counter()
    label = _heuristic(state["query"])
    tokens_in = tokens_out = 0
    try:
        content, tokens_in, tokens_out = groq_client.chat_completion(
            model=settings.groq_model_simple,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify the user question as exactly one word: "
                        "simple, multi_hop, or needs_table. No explanation."
                    ),
                },
                {"role": "user", "content": state["query"]},
            ],
            max_tokens=5,
        )
        cleaned = content.strip().lower()
        if cleaned in {"simple", "multi_hop", "needs_table"}:
            label = cleaned
    except Exception:
        pass  # heuristic label already set; router must never crash the graph

    model = settings.groq_model_simple if label == "simple" else settings.groq_model_complex
    state["model"] = model
    state["token_budget_left"] -= tokens_in + tokens_out

    duration_ms = int((time.perf_counter() - t0) * 1000)
    state["trace"].record_step(
        "router", duration_ms, tokens_in, tokens_out, label=label, model=model
    )
    state["trace"].model_path.append(model)
    return state
