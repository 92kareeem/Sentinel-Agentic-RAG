"""Graph routing decisions, tested as pure functions.

The conditional-edge functions in graph.py encode the system's cost and
latency behaviour, but they are the part least covered by end-to-end tests
(reaching a specific branch through a live graph needs the LLM to cooperate).
They are pure functions of state, so they are tested directly here.
"""

from __future__ import annotations

import time
from typing import Any

import pytest
from app.agents.graph import (
    _route_after_critic,
    _route_after_grounding,
    _route_after_synthesizer,
    refusal_node,
)
from app.config import get_settings
from app.models.schemas import CriticScores, RefusalReason, is_insufficient_context
from app.observability.tracing import TraceRecorder


def _state(**over: Any) -> Any:
    base: dict[str, Any] = {
        "query": "q",
        "search_query": "q",
        "user_id": "u",
        "doc_id": None,
        "trace": TraceRecorder(),
        "attempt": 0,
        "model": "m",
        "token_budget_left": 10_000,
        "deadline_ts": time.monotonic() + 60,
        "retrieved": [],
        "answer": "An answer. [chunk:a]",
        "citations": [],
        "critic": None,
        "status": "running",
        "refusal_reason": None,
        "conversation_history": [],
    }
    base.update(over)
    return base


# ------------------------------------------------- the refusal short-circuit


def test_refusal_on_first_attempt_gets_one_rewrite_not_the_critic() -> None:
    """A rewrite re-runs RETRIEVAL, so it can genuinely recover an answer that
    the first retrieval missed. That round is worth paying for."""
    assert _route_after_synthesizer(
        _state(answer="INSUFFICIENT_CONTEXT", attempt=0)
    ) == "repair_rewrite"


def test_refusal_after_a_rewrite_goes_straight_to_refusal_never_escalates() -> None:
    """Escalation re-runs the bigger model against the SAME evidence. If the
    evidence lacks the answer, a larger model agreeing costs ~15s for nothing,
    and a larger model DISagreeing means it invented an unsupported claim.

    Regression: this path used to run critic -> repair_rewrite ->
    repair_escalate -> critic -> refusal, which was the system's entire p95
    (~25s vs a ~9.5s p50 on the eval set).
    """
    assert _route_after_synthesizer(
        _state(answer="INSUFFICIENT_CONTEXT", attempt=1)
    ) == "refusal"


def test_refusal_short_circuit_survives_llm_decoration() -> None:
    """The model does not always emit the bare token; an exact == comparison
    (what every call site used to do) misses these and pays the full loop."""
    for decorated in [
        "INSUFFICIENT_CONTEXT.",
        "  insufficient_context  ",
        '"INSUFFICIENT_CONTEXT"',
        "INSUFFICIENT_CONTEXT [chunk:a1b2]",
    ]:
        assert is_insufficient_context(decorated), decorated
        assert _route_after_synthesizer(_state(answer=decorated, attempt=1)) == "refusal"


def test_a_real_answer_mentioning_the_token_is_not_treated_as_a_refusal() -> None:
    answer = "The policy returns INSUFFICIENT_CONTEXT when evidence is missing. [chunk:a]"
    assert not is_insufficient_context(answer)
    assert _route_after_synthesizer(_state(answer=answer)) == "critic"


def test_a_normal_answer_still_reaches_the_critic() -> None:
    assert _route_after_synthesizer(_state()) == "critic"


def test_refusal_short_circuit_respects_the_repair_budget() -> None:
    """Below the reserve there is no affordable rewrite, so refuse immediately
    rather than starting a round that cannot finish."""
    broke = _state(
        answer="INSUFFICIENT_CONTEXT",
        attempt=0,
        token_budget_left=get_settings().min_repair_token_reserve - 1,
    )
    assert _route_after_synthesizer(broke) == "refusal"


def test_hard_cap_still_wins_over_the_short_circuit() -> None:
    assert _route_after_synthesizer(
        _state(answer="INSUFFICIENT_CONTEXT", status="refused")
    ) == "refusal"


# ------------------------------------------------------------ existing gates


@pytest.mark.parametrize(
    ("faith", "relevance", "expected"),
    [
        (0.9, 0.9, "grounding_check"),
        (0.5, 0.9, "repair_rewrite"),
        (0.9, 0.5, "repair_rewrite"),
    ],
)
def test_critic_thresholds(faith: float, relevance: float, expected: str) -> None:
    state = _state(critic=CriticScores(faithfulness=faith, relevance=relevance))
    assert _route_after_critic(state) == expected


def test_critic_failure_escalates_on_the_second_attempt_then_refuses() -> None:
    low = CriticScores(faithfulness=0.1, relevance=0.1)
    assert _route_after_critic(_state(critic=low, attempt=1)) == "repair_escalate"
    assert _route_after_critic(_state(critic=low, attempt=2)) == "refusal"


def test_grounding_pass_ends_the_graph() -> None:
    assert _route_after_grounding(_state(status="answered")) == "end"


def test_grounding_failure_repairs_then_refuses() -> None:
    assert _route_after_grounding(_state(status="running", attempt=0)) == "repair_rewrite"
    assert _route_after_grounding(_state(status="running", attempt=1)) == "repair_escalate"
    assert _route_after_grounding(_state(status="running", attempt=2)) == "refusal"


# ------------------------------------------------------ refusal classification


def test_budget_refusal_is_not_reported_as_missing_evidence() -> None:
    """These must stay distinct: a budget refusal is retryable and says nothing
    about whether the corpus covers the question, while INSUFFICIENT_EVIDENCE
    tells the user to rephrase or upload something. Conflating them sends the
    user to fix the wrong thing."""
    state = _state(answer="", token_budget_left=0, status="refused")
    assert refusal_node(state)["refusal_reason"] == RefusalReason.BUDGET_EXHAUSTED

    expired = _state(answer="", deadline_ts=time.monotonic() - 1, status="refused")
    assert refusal_node(expired)["refusal_reason"] == RefusalReason.BUDGET_EXHAUSTED


def test_sentinel_answer_is_classified_as_insufficient_evidence() -> None:
    state = _state(answer="INSUFFICIENT_CONTEXT", status="refused")
    assert refusal_node(state)["refusal_reason"] == RefusalReason.INSUFFICIENT_EVIDENCE


def test_drafted_but_unverifiable_answer_is_its_own_reason() -> None:
    """There WAS evidence and an answer — it just failed the critic or the
    grounding gate. Telling the user "not in your documents" would be wrong."""
    state = _state(answer="Refunds take 30 days. [chunk:a]", status="refused")
    assert refusal_node(state)["refusal_reason"] == RefusalReason.UNVERIFIABLE_ANSWER


def test_refusal_always_leaves_a_user_facing_message() -> None:
    assert refusal_node(_state(answer=""))["answer"].strip()


def test_no_repair_path_can_loop_forever() -> None:
    """Every repair increments `attempt`, and attempt >= 2 refuses everywhere."""
    for route in (_route_after_critic, _route_after_grounding):
        state = _state(critic=CriticScores(faithfulness=0.0, relevance=0.0), attempt=2)
        state["status"] = "running"
        assert route(state) == "refusal"
