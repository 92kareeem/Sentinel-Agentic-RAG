"""LangGraph wiring: router -> retriever -> synthesizer -> critic -> repair loop.

Role in architecture: the single source of truth for how a request flows
through the agent. Every routing decision (pass, rewrite, escalate, refuse)
lives in the conditional edges below, not scattered across node files — the
whole self-healing state machine is readable in one place.
"""

from collections.abc import Callable
from typing import Any

from langgraph.graph import END, StateGraph

from app.agents import critic as critic_mod
from app.agents import repair, retriever, router, synthesizer
from app.agents.budget import check_budget
from app.agents.state import AgentState
from app.config import get_settings
from app.models.schemas import RefusalReason, is_insufficient_context


def grounding_check_node(state: AgentState) -> AgentState:
    """Deterministic grounding gate (guardrails/grounding.py).

    Strips unsupported sentences; if more than MAX_STRIPPED_RATIO of them were
    stripped the answer is untrustworthy — treated like a critic failure so the
    repair loop (or refusal) takes over rather than shipping a
    hallucination-heavy answer. (This docstring previously named a hardcoded
    30%; the threshold is defined once in grounding.py and is currently 50%.)
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
    """Single funnel for every refusal, and the one place they are classified.

    The reason is derived here from observable state rather than being threaded
    through the six nodes that can trigger a refusal: each of those already
    sets status="refused" and returns, and the conditions that distinguish the
    cases are stable by the time this runs (a passed deadline stays passed, an
    exhausted token budget stays exhausted). One classifier is easier to keep
    correct than six assignment sites that must not drift apart.
    """
    state["status"] = "refused"

    if not check_budget(state):
        # A hard cap fired before the graph could finish — the corpus may well
        # contain the answer, so this must not be reported as "not in your
        # documents". It is retryable; INSUFFICIENT_EVIDENCE is not.
        state["refusal_reason"] = RefusalReason.BUDGET_EXHAUSTED
    elif is_insufficient_context(state.get("answer", "")):
        # The model read the evidence and said it doesn't answer the question.
        state["refusal_reason"] = RefusalReason.INSUFFICIENT_EVIDENCE
    else:
        # An answer was drafted but could not be verified — it failed the
        # critic or the grounding gate. Distinct from "no evidence": there WAS
        # evidence, we just couldn't stand behind what was written from it.
        state["refusal_reason"] = RefusalReason.UNVERIFIABLE_ANSWER

    if not state.get("answer"):
        state["answer"] = "I don't have enough verified context to answer this confidently."
    return state


def _guard(next_node: str) -> Callable[[AgentState], str]:
    """Route to refusal instead of next_node if the entering node hit a hard cap."""

    def _route(state: AgentState) -> str:
        return "refusal" if state["status"] == "refused" else next_node

    return _route


def _can_afford_repair(state: AgentState) -> bool:
    """A rewrite or escalate round re-runs retriever+synthesizer+critic — a
    full token-cost pass, not a cheap patch. Below the reserve, that attempt
    would start, spend tokens, and still fail check_budget() partway through
    (most likely inside critic, after synthesizer already spent the bulk of
    it) — strictly worse than refusing now: same outcome, wasted latency and
    quota on a call that could never finish. Route straight to refusal.
    """
    return state["token_budget_left"] >= get_settings().min_repair_token_reserve


def _route_after_synthesizer(state: AgentState) -> str:
    """Short-circuit an honest refusal instead of paying the repair loop for it.

    When the synthesizer answers INSUFFICIENT_CONTEXT it has already reached
    the correct outcome. Sending that to the critic guarantees a failure:
    the critic scores `relevance = does the answer address the question?`, and
    a refusal addresses nothing, so relevance ~ 0 routes it into repair. The
    measured cost of that was the system's entire p95 — the three unanswerable
    eval questions each ran 2 repairs and took ~25s against a p50 of ~9.5s, to
    re-derive a refusal produced in the first round.

    One rewrite is still allowed, because repair_rewrite reformulates the query
    and RE-RUNS RETRIEVAL: "the evidence isn't here" may really mean "retrieval
    missed it", and a second look can legitimately recover an answer. That is
    the round worth paying for, and keeping it is why this change does not
    trade latency for false refusals.

    Escalation is never worth paying for here: repair_escalate only swaps in
    the bigger model against the SAME evidence set. If the small model reports
    the evidence doesn't contain the answer, a larger model reading identical
    text will almost always agree — and in the rare case it disagrees, it has
    talked itself into a claim the evidence didn't support, which is precisely
    what this system exists to prevent.
    """
    if state["status"] == "refused":
        return "refusal"
    if not is_insufficient_context(state["answer"]):
        return "critic"
    if state["attempt"] == 0 and _can_afford_repair(state):
        return "repair_rewrite"
    return "refusal"


def _route_after_critic(state: AgentState) -> str:
    if state["status"] == "refused":
        return "refusal"
    c = state["critic"]
    if c and c.faithfulness >= 0.7 and c.relevance >= 0.7:
        return "grounding_check"
    if not _can_afford_repair(state):
        return "refusal"
    if state["attempt"] == 0:
        return "repair_rewrite"
    if state["attempt"] == 1:
        return "repair_escalate"
    return "refusal"


def _route_after_grounding(state: AgentState) -> str:
    if state["status"] == "answered":
        return "end"
    if not _can_afford_repair(state):
        return "refusal"
    if state["attempt"] == 0:
        return "repair_rewrite"
    if state["attempt"] == 1:
        return "repair_escalate"
    return "refusal"


def build_graph() -> Any:
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
        "synthesizer",
        _route_after_synthesizer,
        {
            "critic": "critic",
            "repair_rewrite": "repair_rewrite",
            "refusal": "refusal",
        },
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