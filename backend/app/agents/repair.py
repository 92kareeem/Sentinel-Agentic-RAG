"""Repair nodes: query rewrite (attempt 0->1) and model escalation (1->2).

Role in architecture: the "self-healing" half of the graph. A failed critic
score doesn't end the request — it triggers one of two repair strategies
depending on how many attempts have already been spent, per the routing
table in graph.py.
"""

import time

from app.agents.budget import check_budget
from app.agents.state import AgentState
from app.config import get_settings
from app.llm import groq_client

_REWRITE_PROMPT = (
    "Rewrite the user's question to be more specific and retrieval-friendly, "
    "using terms likely to appear in source documents. Return ONLY the rewritten "
    "question, no explanation."
)


def repair_rewrite_node(state: AgentState) -> AgentState:
    if not check_budget(state):
        state["status"] = "refused"
        return state

    settings = get_settings()
    t0 = time.perf_counter()
    content, tokens_in, tokens_out = groq_client.chat_completion(
        model=settings.groq_model_simple,
        messages=[
            {"role": "system", "content": _REWRITE_PROMPT},
            # Always the user's original wording — rewriting a rewrite
            # compounds drift away from what was actually asked.
            {"role": "user", "content": state["query"]},
        ],
        max_tokens=100,
    )
    # Writes search_query, never query. Overwriting the user's question here
    # made every downstream node answer and grade the reformulation instead:
    # the rewrite is tuned for keyword recall ("annual subscription refund
    # window"), so nuance the user cared about ("...if onboarding has already
    # started") was dropped from the question being answered, not just from
    # the one being searched.
    state["search_query"] = content.strip() or state["search_query"]
    state["attempt"] += 1
    state["token_budget_left"] -= tokens_in + tokens_out

    duration_ms = int((time.perf_counter() - t0) * 1000)
    state["trace"].repair_count += 1
    state["trace"].record_step(
        "repair_rewrite", duration_ms, tokens_in, tokens_out, new_query=state["search_query"]
    )
    return state


def repair_escalate_node(state: AgentState) -> AgentState:
    if not check_budget(state):
        state["status"] = "refused"
        return state

    settings = get_settings()
    t0 = time.perf_counter()
    state["model"] = settings.groq_model_complex
    state["attempt"] += 1
    state["trace"].model_path.append(state["model"])

    duration_ms = int((time.perf_counter() - t0) * 1000)
    state["trace"].repair_count += 1
    state["trace"].record_step("repair_escalate", duration_ms, model=state["model"])
    return state
