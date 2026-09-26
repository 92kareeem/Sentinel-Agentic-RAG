"""Router node: picks the Groq model for this query.

Role in architecture: the cheapest node in the graph. It classifies the query
with a keyword heuristic and selects the small or large model accordingly.

WHY THERE IS NO LLM CALL HERE ANY MORE
--------------------------------------
This node used to spend one LLM call per query on the same classification.
Two measurements retired it (harness: `evals/router_ab.py`, full comparison in
ADR 0002):

1. It had never actually worked. `max_tokens=20` was entirely consumed by
   gpt-oss's hidden reasoning tokens, so the classifier returned empty content
   on 18-20 of 20 golden questions, silently fell through to this heuristic,
   and still charged ~385ms and its tokens to every request.

2. Repaired (max_tokens=120) and measured against the heuristic on the same
   corpus and judge, it was worse: mean faithfulness 0.955 -> 0.925, mean
   tokens/query 1,609 -> 1,838, and it introduced a false refusal — golden
   question 2 was labelled `needs_table`, escalated to the 120b model, and
   refused a question the 20b model answers correctly in 1.5s.

The burden of proof is on keeping an LLM call, not on removing one. With no
measured upside and a measured cost in latency, tokens and one false refusal,
the heuristic stands alone. Re-run `evals/router_ab.py` before reintroducing
it — the harness is kept precisely so this stays a measurable decision rather
than a matter of taste.
"""

import time

from app.agents.budget import check_budget
from app.agents.state import AgentState
from app.config import get_settings

# Terms that suggest the question spans multiple facts or a table, which is
# where the larger model earns its cost.
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
    model = settings.groq_model_simple if label == "simple" else settings.groq_model_complex
    state["model"] = model

    duration_ms = int((time.perf_counter() - t0) * 1000)
    # `source` is recorded even though there is only one source today: a trace
    # that reports a label without reporting who produced it is how the old
    # LLM classifier managed to be completely broken while looking healthy.
    state["trace"].record_step(
        "router", duration_ms, 0, 0, label=label, model=model, source="heuristic"
    )
    state["trace"].model_path.append(model)
    return state
