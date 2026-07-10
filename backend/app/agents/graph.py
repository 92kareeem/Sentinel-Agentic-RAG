"""LangGraph wiring: router -> retriever -> synthesizer -> critic -> repair loop.

Role in architecture: the single source of truth for how a request flows
through the agent. Every routing decision (pass, rewrite, escalate, refuse)
lives in the conditional edges below, not scattered across node files — the
whole self-healing state machine is readable in one place.
"""

from langgraph.graph import END, StateGraph

from app.agents import repair, retriever, router, synthesizer
from app.agents import critic as critic_mod
from app.agents.state import AgentState


def grounding_check_node(state: AgentState) -> AgentState:
    """Deterministic citation check (P2 stub of guardrails/grounding.py; P3 hardens it).

    Every cited chunk_id must exist among the retrieved chunks — refuses
    rather than trusting the critic's score blindly if the answer cites
    chunks that were never actually retrieved.
    """
    retrieved_ids = {c.chunk_id for c in state["retrieved"]}
    cited_ids = {c.chunk_id for c in state["citations"]}
    if cited_ids and not cited_ids.issubset(retrieved_ids):
        state["status"] = "refused"
        state["answer"] = "Answer referenced chunks outside the retrieved context; refusing."
        return state
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
    g.add_conditional_edges("router", _guard("retriever"), {"retriever": "retriever", "refusal": "refusal"})
    g.add_conditional_edges(
        "retriever", _guard("synthesizer"), {"synthesizer": "synthesizer", "refusal": "refusal"}
    )
    g.add_conditional_edges("synthesizer", _guard("critic"), {"critic": "critic", "refusal": "refusal"})
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
    g.add_edge("grounding_check", END)
    g.add_edge("refusal", END)
    return g.compile()
