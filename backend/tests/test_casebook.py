"""The casebook: turning failures into a document owner's work list.

The tests that matter here are about business meaning, not storage. A gap
report that miscounts, misclassifies, or reports a scary-looking answer rate
is worse than no report: someone will act on it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from app.config import get_settings
from app.learning import casebook
from app.models.schemas import Case, CaseDiagnosis, CaseOutcome


@pytest.fixture()
def store(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("INDEX_DIR", str(tmp_path))
    monkeypatch.setenv("LOCAL_MODE", "true")
    get_settings.cache_clear()
    casebook.reset_local()
    yield tmp_path
    get_settings.cache_clear()


def _case(question: str, diagnosis: CaseDiagnosis, *, days_ago: int = 0, owner: str = "u") -> Case:
    when = datetime.now(UTC) - timedelta(days=days_ago)
    return Case(
        case_id=casebook.new_case_id(),
        owner_id=owner,
        trace_id="t",
        question=question,
        outcome=(
            CaseOutcome.ANSWERED
            if diagnosis == CaseDiagnosis.CITATION_MISATTRIBUTED
            else CaseOutcome.REFUSED
        ),
        diagnosis=diagnosis,
        created_at=when.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


# ------------------------------------------------------------------ diagnosis


def test_nothing_retrieved_is_reported_as_a_documentation_gap() -> None:
    assert (
        casebook.diagnose(
            outcome=CaseOutcome.REFUSED,
            refusal_reason="INSUFFICIENT_EVIDENCE",
            retrieved_chunks=0,
            repaired_citations=0,
        )
        == CaseDiagnosis.NO_EVIDENCE_FOUND
    )


def test_evidence_found_but_unusable_is_a_different_gap() -> None:
    """Distinct from NO_EVIDENCE_FOUND because the fix is different: the topic
    is covered, the specific case is not."""
    assert (
        casebook.diagnose(
            outcome=CaseOutcome.REFUSED,
            refusal_reason="INSUFFICIENT_EVIDENCE",
            retrieved_chunks=5,
            repaired_citations=0,
        )
        == CaseDiagnosis.EVIDENCE_OFF_TOPIC
    )


def test_a_failed_verification_is_blamed_on_the_system_not_the_documents() -> None:
    """The evidence was there and we still could not stand behind an answer.
    Telling an owner to write another document would be wrong advice."""
    assert (
        casebook.diagnose(
            outcome=CaseOutcome.REFUSED,
            refusal_reason="UNVERIFIABLE_ANSWER",
            retrieved_chunks=4,
            repaired_citations=0,
        )
        == CaseDiagnosis.ANSWER_UNVERIFIABLE
    )


def test_a_budget_refusal_is_never_reported_as_a_knowledge_gap() -> None:
    """A cap firing says nothing about the corpus. Counting it as a gap sends
    someone to write a document that would not have helped."""
    diagnosis = casebook.diagnose(
        outcome=CaseOutcome.REFUSED,
        refusal_reason="BUDGET_EXHAUSTED",
        retrieved_chunks=0,
        repaired_citations=0,
    )
    assert diagnosis == CaseDiagnosis.CAPACITY_EXCEEDED
    assert diagnosis not in casebook.GAP_DIAGNOSES


def test_a_clean_answer_records_nothing() -> None:
    """Otherwise the casebook is a query log and the gap report is unreadable."""
    assert (
        casebook.diagnose(
            outcome=CaseOutcome.ANSWERED,
            refusal_reason=None,
            retrieved_chunks=5,
            repaired_citations=0,
        )
        is None
    )


def test_an_answer_whose_citations_needed_repair_is_still_recorded() -> None:
    """It shipped, so it is not a failure — but it is the only signal anyone
    gets that the corpus has near-duplicate passages."""
    assert (
        casebook.diagnose(
            outcome=CaseOutcome.ANSWERED,
            refusal_reason=None,
            retrieved_chunks=5,
            repaired_citations=2,
        )
        == CaseDiagnosis.CITATION_MISATTRIBUTED
    )


# ------------------------------------------------------------------ the report


def test_the_same_question_asked_different_ways_is_one_gap(store: Any) -> None:
    """THE point of clustering. Six people asking the same thing is one
    missing paragraph, and a flat list of near-duplicates is a report nobody
    reads twice."""
    for q in [
        "What is the refund window for annual subscriptions?",
        "How long do customers have to request a refund on an annual plan?",
        "refund window annual subscription",
    ]:
        casebook.record(_case(q, CaseDiagnosis.NO_EVIDENCE_FOUND))

    report = casebook.knowledge_gaps("u")

    assert len(report.gaps) == 1
    assert report.gaps[0].question_count == 3


def test_unrelated_questions_stay_separate_gaps(store: Any) -> None:
    casebook.record(_case("What is the refund window?", CaseDiagnosis.NO_EVIDENCE_FOUND))
    casebook.record(
        _case("How much parental leave do contractors get?", CaseDiagnosis.NO_EVIDENCE_FOUND)
    )

    assert len(casebook.knowledge_gaps("u").gaps) == 2


def test_gaps_are_ranked_by_how_often_they_were_asked(store: Any) -> None:
    """An owner's prioritisation question is "what is costing us most time?",
    not "what happened most recently?"."""
    casebook.record(_case("vpn access request process", CaseDiagnosis.NO_EVIDENCE_FOUND))
    for i in range(4):
        casebook.record(
            _case(f"expense reimbursement limit for travel {i}", CaseDiagnosis.NO_EVIDENCE_FOUND)
        )

    gaps = casebook.knowledge_gaps("u").gaps

    assert gaps[0].question_count == 4
    assert "expense" in gaps[0].topic


def test_every_gap_carries_an_action_a_non_engineer_can_take(store: Any) -> None:
    casebook.record(_case("what is the refund window", CaseDiagnosis.NO_EVIDENCE_FOUND))

    gap = casebook.knowledge_gaps("u").gaps[0]

    assert gap.recommended_action == casebook.RECOMMENDED_ACTION[CaseDiagnosis.NO_EVIDENCE_FOUND]
    assert gap.example_questions


def test_system_failures_are_recorded_but_are_not_documentation_gaps(store: Any) -> None:
    """ANSWER_UNVERIFIABLE and CAPACITY_EXCEEDED are our problems. Putting them
    on the owner's list would be passing the buck."""
    casebook.record(_case("refund window", CaseDiagnosis.ANSWER_UNVERIFIABLE))
    casebook.record(_case("notice period", CaseDiagnosis.CAPACITY_EXCEEDED))

    report = casebook.knowledge_gaps("u")

    assert report.gaps == []
    assert len(casebook.list_for_owner("u")) == 2


def test_cases_outside_the_window_are_excluded(store: Any) -> None:
    casebook.record(
        _case("old question about refunds", CaseDiagnosis.NO_EVIDENCE_FOUND, days_ago=60)
    )
    casebook.record(_case("recent question about leave", CaseDiagnosis.NO_EVIDENCE_FOUND))

    assert len(casebook.knowledge_gaps("u", window_days=30).gaps) == 1
    assert len(casebook.knowledge_gaps("u", window_days=90).gaps) == 2


def test_one_tenants_questions_never_appear_in_anothers_report(store: Any) -> None:
    """The questions people ask are at least as sensitive as the documents
    they ask about."""
    casebook.record(
        _case(
            "acquisition due diligence checklist",
            CaseDiagnosis.NO_EVIDENCE_FOUND,
            owner="alice",
        )
    )
    casebook.record(_case("holiday policy", CaseDiagnosis.NO_EVIDENCE_FOUND, owner="bob"))

    bob = casebook.knowledge_gaps("bob")

    assert len(bob.gaps) == 1
    assert "acquisition" not in bob.gaps[0].topic


# ------------------------------------------------------------- the answer rate


def test_the_answer_rate_counts_successes_that_were_never_recorded_as_cases(
    store: Any,
) -> None:
    """THE regression test for the headline number.

    The casebook holds failures only. Deriving the rate from it would report
    something near zero and make a healthy system look broken, so volume is
    counted separately.
    """
    for _ in range(9):
        casebook.bump_totals("u", CaseOutcome.ANSWERED)
    casebook.bump_totals("u", CaseOutcome.REFUSED)
    casebook.record(_case("parental leave for contractors", CaseDiagnosis.NO_EVIDENCE_FOUND))

    report = casebook.knowledge_gaps("u")

    assert report.total_questions == 10
    assert report.answered == 9
    assert report.unanswered == 1
    assert report.answer_rate == 0.9


def test_a_workspace_that_has_never_been_queried_reports_no_failures(store: Any) -> None:
    """A fresh workspace must not render as 0% — an empty denominator is not
    a bad score."""
    report = casebook.knowledge_gaps("nobody")

    assert report.total_questions == 0
    assert report.answer_rate == 1.0
    assert report.gaps == []


# --------------------------------------------------------------- durability


def test_a_failed_case_write_never_breaks_the_request(
    store: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A user who asked a hard question must not lose their answer BECAUSE it
    was hard."""

    def boom(*_: Any, **__: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(casebook, "_local_write_all", boom)

    casebook.record(_case("anything", CaseDiagnosis.NO_EVIDENCE_FOUND))  # must not raise


def test_a_corrupt_casebook_reads_as_empty_rather_than_failing(store: Any) -> None:
    casebook.record(_case("refund window", CaseDiagnosis.NO_EVIDENCE_FOUND))
    casebook._local_path().write_text("{not json", encoding="utf-8")

    assert casebook.list_for_owner("u") == []
