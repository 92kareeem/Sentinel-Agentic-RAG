"""Two ways the pipeline could deliver something it never stood behind.

Both are silent: nothing raises, nothing is logged, and the response looks
well-formed. That is what makes them worth pinning with tests — neither would
be noticed in a demo, and both directly contradict the product's one promise.
"""

from __future__ import annotations

import time
from typing import Any

from app.agents.graph import refusal_node
from app.agents.repair import repair_rewrite_node
from app.models.schemas import RefusalReason, safe_refusal_text
from app.observability.tracing import TraceRecorder

REJECTED_DRAFT = (
    "Annual subscriptions are refundable for 45 days with no conditions. [chunk:a]"
)


def _state(**over: Any) -> Any:
    base: dict[str, Any] = {
        "query": "Can we refund an annual plan after 20 days if onboarding started?",
        "search_query": "Can we refund an annual plan after 20 days if onboarding started?",
        "user_id": "u",
        "doc_id": None,
        "trace": TraceRecorder(),
        "attempt": 0,
        "model": "m",
        "token_budget_left": 10_000,
        "deadline_ts": time.monotonic() + 60,
        "retrieved": [],
        "answer": "",
        "citations": [],
        "critic": None,
        "status": "running",
        "refusal_reason": None,
        "conversation_history": [],
    }
    base.update(over)
    return base


# --------------------------------------------- a refusal must not ship the draft


def test_refusal_never_returns_the_draft_it_just_rejected() -> None:
    """THE regression test.

    A draft that failed the critic or the grounding gate used to survive in
    state["answer"], and the API returned that field as RefusalResponse.reason.
    The unverified claim reached the user anyway — with the refusal acting as a
    disclaimer in front of it rather than as a gate.
    """
    out = refusal_node(_state(answer=REJECTED_DRAFT))

    assert "45 days" not in out["answer"]
    assert REJECTED_DRAFT not in out["answer"]
    assert out["answer"] == safe_refusal_text(RefusalReason.UNVERIFIABLE_ANSWER)


def test_the_rejected_draft_is_kept_for_diagnosis_but_off_the_response_path() -> None:
    """Discarding it entirely would be safe but would throw away the best
    signal for triage: what the model *wanted* to say separates a retrieval
    miss from a synthesis failure."""
    out = refusal_node(_state(answer=REJECTED_DRAFT))

    assert out["rejected_draft"] == REJECTED_DRAFT
    assert out["rejected_draft"] != out["answer"]


def test_refusal_drops_citations_so_none_render_beneath_it() -> None:
    """Citations described claims in a draft that is no longer being made;
    showing them under a refusal implies the refusal was evidence-backed."""
    from app.models.schemas import Citation

    cites = [Citation(chunk_id="a", section_path="Refunds", quote="...", document_id="d")]
    assert refusal_node(_state(answer=REJECTED_DRAFT, citations=cites))["citations"] == []


def test_an_honest_insufficient_context_is_not_recorded_as_a_rejected_draft() -> None:
    """The sentinel is a correct outcome, not a suppressed answer — recording
    it as a rejected draft would flood triage with successes."""
    out = refusal_node(_state(answer="INSUFFICIENT_CONTEXT"))

    assert "rejected_draft" not in out
    assert out["refusal_reason"] == RefusalReason.INSUFFICIENT_EVIDENCE


def test_every_reason_code_has_safe_text_and_none_of_it_is_model_output() -> None:
    for reason in RefusalReason:
        text = safe_refusal_text(reason)
        assert text and not text.startswith("INSUFFICIENT_CONTEXT")


def test_budget_exhaustion_outranks_a_failed_draft() -> None:
    """A cap firing says nothing about whether the documents cover the
    question, and it is retryable — reporting UNVERIFIABLE_ANSWER would send
    the user to fix their documents instead of pressing retry."""
    out = refusal_node(_state(answer=REJECTED_DRAFT, token_budget_left=0))
    assert out["refusal_reason"] == RefusalReason.BUDGET_EXHAUSTED


# ------------------------------------- a rewrite must not replace the question


def test_repair_rewrite_changes_the_search_query_not_the_users_question(
    monkeypatch: Any,
) -> None:
    """THE regression test for intent drift.

    The rewrite is tuned for keyword recall, so it drops qualifiers the user
    cared about. When it overwrote state["query"], the synthesizer answered
    the reformulation and the critic scored relevance against it too: every
    stage agreed on a well-cited answer to a question nobody asked.
    """
    from app.agents import repair

    monkeypatch.setattr(
        repair.groq_client,
        "chat_completion",
        lambda **_: ("annual subscription refund window policy", 10, 5),
    )

    state = _state()
    original = state["query"]
    out = repair_rewrite_node(state)

    assert out["query"] == original, "the user's question was overwritten"
    assert out["search_query"] == "annual subscription refund window policy"
    assert "onboarding" in out["query"]


def test_a_rewrite_rewrites_the_original_question_not_the_previous_rewrite(
    monkeypatch: Any,
) -> None:
    """Chaining rewrites compounds drift; each one starts from what was asked."""
    from app.agents import repair

    seen: list[str] = []

    def fake(**kw: Any) -> tuple[str, int, int]:
        seen.append(kw["messages"][-1]["content"])
        return ("reformulated", 10, 5)

    monkeypatch.setattr(repair.groq_client, "chat_completion", fake)

    state = _state(search_query="an earlier reformulation")
    repair_rewrite_node(state)

    assert seen == [state["query"]]


def test_an_empty_rewrite_falls_back_without_clobbering_the_search_query(
    monkeypatch: Any,
) -> None:
    from app.agents import repair

    monkeypatch.setattr(repair.groq_client, "chat_completion", lambda **_: ("   ", 5, 1))

    out = repair_rewrite_node(_state(search_query="prior search terms"))
    assert out["search_query"] == "prior search terms"
