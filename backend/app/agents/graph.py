"""LangGraph wiring: router -> retriever -> synthesizer -> critic -> repair loop.

Role in architecture: the single source of truth for how a request flows
through the agent. Every routing decision (pass, rewrite, escalate, refuse)
lives in the conditional edges below, not scattered across node files — the
whole self-healing state machine is readable in one place.
"""

from langgraph.graph import END, StateGraph

from app.agents import critic as critic_mod
from app.agents import repair, retriever, router, synthesizer
from app.agents.state import AgentState


def grounding_check_node(state: AgentState) -> AgentState:
    """Deterministic grounding gate (guardrails/grounding.py).

    Strips unsupported sentences; if >30% were stripped the answer is
    untrustworthy — treated like a critic failure so the repair loop (or
    refusal) takes over rather than shipping a hallucination-heavy answer.
    """
    from app.guardrails import grounding

    result = grounding.verify(state["answer"], state["retrieved"])
    state["trace"].record_step(
        "grounding_check", 0, stripped_ratio=f"{result.stripped_ratio:.2f}", ok=result.ok
    )
    if not result.ok:
        state["status"] = "running"  # routed like a critic failure below
        return state
    state["answer"] = result.clean_answer
    valid = set(result.valid_chunk_ids)
    state["citations"] = [c for c in state["citations"] if c.chunk_id in valid]
    state["status"] = "answered"
    return state


def refusal_node(state: AgentState) -> AgentState:
    state["status"] = "refused"
    if not state.get("answer"):
        state["answer"] = "I don't have enough verified context to answer this confidently."
    return state


def _guard(next_node: str):
    """Route to refusal instead of next_node if the entering node hit a hard cap."""

    def _route(state: AgentState) -> str:
        return "refusal" if state["status"] == "refused" else next_node

    return _route


def _route_after_critic(state: AgentState) -> str:
    if state["status"] == "refused":
        return "refusal"
    c = state["critic"]
    if c and c.faithfulness >= 0.7 and c.relevance >= 0.7:
        return "grounding_check"
    if state["attempt"] == 0:
        return "repair_rewrite"
    if state["attempt"] == 1:
        return "repair_escalate"
    return "refusal"


def _route_after_grounding(state: AgentState) -> str:
    if state["status"] == "answered":
        return "end"
    if state["attempt"] == 0:
        return "repair_rewrite"
    if state["attempt"] == 1:
        return "repair_escalate"
    return "refusal"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("router", router.router_node)
    g.add_node("retriever", retriever.retriever_node)
    g.add_node("synthesizer", synthesizer.synthesizer_node)
    g.add_node("critic", critic_mod.critic_node)
    g.add_node("repair_rewrite", repair.repair_rewrite_node)
    g.add_node("repair_escalate", repair.repair_escalate_node)
    g.add_node("grounding_check", grounding_check_node)
    g.add_node("refusal", refusal_node)

    g.set_entry_point("router")
    g.add_conditional_edges(
        "router", _guard("retriever"), {"retriever": "retriever", "refusal": "refusal"}
    )
    g.add_conditional_edges(
        "retriever", _guard("synthesizer"), {"synthesizer": "synthesizer", "refusal": "refusal"}
    )
    g.add_conditional_edges(
        "synthesizer", _guard("critic"), {"critic": "critic", "refusal": "refusal"}
    )
    g.add_conditional_edges(
        "critic",
        _route_after_critic,
        {
            "grounding_check": "grounding_check",
            "repair_rewrite": "repair_rewrite",
            "repair_escalate": "repair_escalate",
            "refusal": "refusal",
        },
    )
    g.add_edge("repair_rewrite", "retriever")
    g.add_edge("repair_escalate", "synthesizer")
    g.add_conditional_edges(
        "grounding_check",
        _route_after_grounding,
        {
            "end": END,
            "repair_rewrite": "repair_rewrite",
            "repair_escalate": "repair_escalate",
            "refusal": "refusal",
        },
    )
    g.add_edge("refusal", END)
    return g.compile()
