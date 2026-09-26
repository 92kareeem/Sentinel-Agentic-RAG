"""The regression ratchet: a fixed mistake must never come back unnoticed.

These are the rules that make the ratchet trustworthy, tested rather than
trusted — no LLM, no network. The LLM only produces the scores; everything
that decides what a score MEANS lives here.

The first test runs against the committed regressions.json, so a hand edit
that deletes a lesson to get CI green fails CI instead.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from app.config import get_settings
from app.learning import casebook, ratchet
from app.models.schemas import Case, CaseDiagnosis, CaseOutcome, FeedbackVerdict

REPO = Path(__file__).resolve().parents[2]
COMMITTED_SUITE = REPO / "evals" / "regressions.json"


def _case(
    case_id: str = "c1",
    diagnosis: CaseDiagnosis = CaseDiagnosis.USER_REPORTED_INCORRECT,
    *,
    correction: str | None = "Customers have 30 days to request a refund.",
    question: str = "How many days do customers have to request a refund?",
) -> Case:
    return Case(
        case_id=case_id,
        owner_id="demo",
        trace_id=f"t-{case_id}",
        question=question,
        outcome=CaseOutcome.ANSWERED,
        diagnosis=diagnosis,
        created_at="2026-09-20T10:00:00Z",
        correction=correction,
    )


def _suite_with(*cases: Case) -> ratchet.RegressionSuite:
    suite = ratchet.RegressionSuite()
    for c in cases:
        ratchet.promote(suite, c)
    return suite


def _row(**kw: Any) -> dict[str, Any]:
    base = {"refused": False, "faithfulness": 1.0, "completeness": 1.0}
    return {**base, **kw}


# ------------------------------------------------------- the committed file


def test_the_committed_regression_suite_is_valid() -> None:
    """The guard on the file itself. Deleting R003 to make a red build green
    leaves a hole in the numbering, and this test is what notices."""
    suite = ratchet.load(COMMITTED_SUITE)
    assert ratchet.validate(suite) == []


# --------------------------------------------------------------- invariants


def test_deleting_a_regression_is_detected() -> None:
    suite = _suite_with(_case("c1"), _case("c2"), _case("c3"))
    del suite.active[1]  # quietly remove R002

    problems = ratchet.validate(suite)

    assert any("R002" in p and "deleted rather than retired" in p for p in problems)


def test_deleting_the_newest_regression_is_detected() -> None:
    """The blind spot of dense numbering alone: removing R003 from R001-R003
    leaves R001, R002 — no gap. It is also the likeliest deletion, because the
    newest lesson is the one still failing. Found during end-to-end testing."""
    suite = _suite_with(_case("c1"), _case("c2"), _case("c3"))
    suite.active.pop()  # quietly remove R003

    problems = ratchet.validate(suite)

    assert any("3 regression id(s) issued but 2 present" in p for p in problems)


def test_an_id_is_not_reused_after_a_deletion() -> None:
    """If a deletion happened anyway, the next id must not recycle the deleted
    one — "R002" in an old report must never mean two different questions."""
    suite = _suite_with(_case("c1"), _case("c2"))
    suite.active.pop()

    assert ratchet.next_id(suite) == "R003"


def test_retiring_a_regression_keeps_the_suite_valid() -> None:
    """The legitimate way out: on the record, with a reason."""
    suite = _suite_with(_case("c1"), _case("c2"), _case("c3"))
    ratchet.retire(suite, "R002", "Refund window changed to 60 days in the 2026 policy.")

    assert ratchet.validate(suite) == []
    assert [i.id for i in suite.active] == ["R001", "R003"]
    assert suite.retired[0].reason.startswith("Refund window changed")


def test_an_answer_regression_without_a_reference_is_invalid() -> None:
    suite = _suite_with(_case())
    suite.active[0].reference_answer = "   "

    assert any("no reference_answer" in p for p in ratchet.validate(suite))


def test_a_refuse_regression_without_a_reason_is_invalid() -> None:
    suite = ratchet.RegressionSuite()
    ratchet.promote(suite, _case(), expect="refuse", note="Payroll data is never in scope.")
    suite.active[0].note = None

    assert any("permanently out of scope" in p for p in ratchet.validate(suite))


def test_enforced_without_a_lock_time_is_invalid() -> None:
    suite = _suite_with(_case())
    suite.active[0].status = "enforced"  # hand-edited, bypassing lock()

    assert any("never locked" in p for p in ratchet.validate(suite))


def test_save_refuses_to_write_a_broken_suite(tmp_path: Path) -> None:
    """The file on disk is always one validate() accepts — so a bug in a
    tool that edits the suite cannot leave CI running against a broken one."""
    suite = _suite_with(_case("c1"), _case("c2"))
    del suite.active[0]
    target = tmp_path / "regressions.json"

    with pytest.raises(ratchet.PromotionError, match="invalid suite"):
        ratchet.save(suite, target)
    assert not target.exists()


def test_save_then_load_round_trips(tmp_path: Path) -> None:
    suite = _suite_with(_case("c1"), _case("c2"))
    ratchet.lock(suite, "R001")
    target = tmp_path / "regressions.json"

    ratchet.save(suite, target)

    assert ratchet.load(target) == suite


# ---------------------------------------------------------------- promotion


def test_a_readers_correction_becomes_the_reference() -> None:
    suite = ratchet.RegressionSuite()
    item = ratchet.promote(suite, _case(correction="It is 30 days."))

    assert item.expect == "answer"
    assert item.reference_answer == "It is 30 days."
    assert item.source.case_id == "c1"
    assert item.source.diagnosis == CaseDiagnosis.USER_REPORTED_INCORRECT


def test_a_reviewers_reference_overrides_the_readers_correction() -> None:
    """Promotion is where a person reviews the correction. A reader can be
    wrong too — if 'refunds are unlimited' went in unreviewed, the suite would
    fail the system for being right."""
    suite = ratchet.RegressionSuite()
    item = ratchet.promote(
        suite,
        _case(correction="Refunds are unlimited."),
        reference="Customers have 30 days to request a refund.",
    )

    assert item.reference_answer == "Customers have 30 days to request a refund."


def test_an_answer_regression_needs_something_to_judge_against() -> None:
    suite = ratchet.RegressionSuite()

    with pytest.raises(ratchet.PromotionError, match="--reference"):
        ratchet.promote(suite, _case(correction=None))
    assert suite.active == []


def test_a_closed_knowledge_gap_is_promoted_as_an_answer() -> None:
    """THE design point. The gap report asks the owner to add the missing
    document. Once they have, the question should be ANSWERED — so a refusal
    case is promoted as `answer`, locking the fix in. Promoting it as
    'must refuse forever' would fail CI for exactly the fix the product asked
    for, putting the ratchet at war with the gap report."""
    suite = ratchet.RegressionSuite()
    refusal = _case(
        "gap1", CaseDiagnosis.NO_EVIDENCE_FOUND, correction=None,
        question="How much parental leave do contractors get?",
    )

    item = ratchet.promote(
        suite, refusal, reference="Contractors get 10 days of parental leave."
    )

    assert item.expect == "answer"
    assert item.source.diagnosis == CaseDiagnosis.NO_EVIDENCE_FOUND


def test_must_refuse_is_never_inferred() -> None:
    """Asserting the documents will NEVER answer a question is a decision.
    Without a reason it could not be told apart from a gap nobody closed."""
    suite = ratchet.RegressionSuite()

    with pytest.raises(ratchet.PromotionError, match="--note"):
        ratchet.promote(suite, _case(), expect="refuse")

    item = ratchet.promote(
        suite, _case(), expect="refuse", note="Salaries are confidential; never in the corpus."
    )
    assert item.expect == "refuse"
    assert item.reference_answer is None


def test_a_capacity_limit_cannot_become_a_regression() -> None:
    """The request ran out of budget before the question got a fair attempt.
    A test built on it would assert a rate limit."""
    suite = ratchet.RegressionSuite()

    with pytest.raises(ratchet.PromotionError, match="capacity"):
        ratchet.promote(suite, _case(diagnosis=CaseDiagnosis.CAPACITY_EXCEEDED))


def test_a_case_cannot_be_promoted_twice() -> None:
    suite = _suite_with(_case("c1"))

    with pytest.raises(ratchet.PromotionError, match="already"):
        ratchet.promote(suite, _case("c1"))


def test_promotion_always_lands_pending() -> None:
    """A case is promoted when someone confirms it is a real failure —
    usually before it is fixed. Enforcing it immediately would turn CI red
    at the moment of reporting, which is exactly when nobody can fix it."""
    item = ratchet.promote(ratchet.RegressionSuite(), _case())

    assert item.status == "pending"
    assert item.locked_at is None


def test_ids_are_never_reused_after_retirement() -> None:
    """'R002 was retired' must always mean the same R002."""
    suite = _suite_with(_case("c1"), _case("c2"))
    ratchet.retire(suite, "R002", "superseded")

    item = ratchet.promote(suite, _case("c3"))

    assert item.id == "R003"
    assert ratchet.validate(suite) == []


# ------------------------------------------------------------ lock / retire


def test_lock_is_one_way() -> None:
    suite = _suite_with(_case())

    item = ratchet.lock(suite, "R001")

    assert item.status == "enforced"
    assert item.locked_at is not None
    with pytest.raises(ratchet.PromotionError, match="already enforced"):
        ratchet.lock(suite, "R001")


def test_retiring_needs_a_reason() -> None:
    suite = _suite_with(_case())

    with pytest.raises(ratchet.PromotionError, match="reason"):
        ratchet.retire(suite, "R001", "   ")
    assert len(suite.active) == 1


def test_a_retired_regression_cannot_be_locked() -> None:
    suite = _suite_with(_case())
    ratchet.retire(suite, "R001", "superseded")

    with pytest.raises(ratchet.PromotionError, match="retired"):
        ratchet.lock(suite, "R001")


# ------------------------------------------------------------------ scoring


def _item(expect: ratchet.Expect = "answer", *, enforced: bool = True) -> ratchet.RegressionItem:
    suite = ratchet.RegressionSuite()
    if expect == "answer":
        item = ratchet.promote(suite, _case())
    else:
        item = ratchet.promote(suite, _case(), expect="refuse", note="out of scope")
    if enforced:
        ratchet.lock(suite, item.id)
    return item


def test_an_upstream_failure_is_never_a_regression() -> None:
    """A rate limit says nothing about whether the mistake came back, and a
    ratchet that cries wolf gets switched off."""
    outcome = ratchet.evaluate(_item(), _row(error="Groq returned 429"))

    assert outcome.passed is None
    assert ratchet.blocking_failures([outcome]) == []


def test_answering_an_out_of_scope_question_fails() -> None:
    assert ratchet.evaluate(_item("refuse"), _row(refused=True)).passed is True
    outcome = ratchet.evaluate(_item("refuse"), _row(refused=False))
    assert outcome.passed is False
    assert "out of scope" in outcome.reason


def test_refusing_a_question_it_learned_to_answer_fails() -> None:
    """The closed-gap loop: once fixed, it may never go back to refusing."""
    outcome = ratchet.evaluate(_item(), _row(refused=True))

    assert outcome.passed is False
    assert "refused" in outcome.reason


def test_an_answer_missing_the_corrected_fact_fails() -> None:
    """The wrong-answer loop: the answer must now contain what the reader
    said it should have. Faithfulness alone would pass a well-grounded
    answer that still says 14 days."""
    outcome = ratchet.evaluate(_item(), _row(faithfulness=1.0, completeness=0.2))

    assert outcome.passed is False
    assert "corrected fact" in outcome.reason


def test_an_ungrounded_answer_fails_even_if_it_has_the_fact() -> None:
    outcome = ratchet.evaluate(_item(), _row(faithfulness=0.3, completeness=1.0))

    assert outcome.passed is False
    assert "not grounded" in outcome.reason


def test_a_grounded_complete_answer_passes() -> None:
    outcome = ratchet.evaluate(_item(), _row(faithfulness=0.9, completeness=0.8))
    assert outcome.passed is True


def test_only_enforced_failures_block() -> None:
    """A pending item is a known failure. Blocking on it would make recording
    a failure and breaking the build the same act."""
    enforced = ratchet.evaluate(_item(enforced=True), _row(refused=True))
    pending = ratchet.evaluate(_item(enforced=False), _row(refused=True))

    assert ratchet.blocking_failures([enforced, pending]) == [enforced]


def test_a_pending_item_that_now_passes_is_ready_to_lock() -> None:
    passing = ratchet.evaluate(_item(enforced=False), _row())
    failing = ratchet.evaluate(_item(enforced=False), _row(refused=True))
    already = ratchet.evaluate(_item(enforced=True), _row())

    assert ratchet.ready_to_lock([passing, failing, already]) == [passing]


# ---------------------------------------------------- the promote.py CLI


@pytest.fixture()
def cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """promote.py over a real local casebook and a scratch suite file."""
    monkeypatch.setenv("INDEX_DIR", str(tmp_path))
    monkeypatch.setenv("LOCAL_MODE", "true")
    get_settings.cache_clear()
    casebook.reset_local()

    spec = importlib.util.spec_from_file_location("promote", REPO / "evals" / "promote.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["promote"] = module
    spec.loader.exec_module(module)

    suite_path = tmp_path / "regressions.json"

    def run(*argv: str) -> int:
        return int(module.main(["--suite", str(suite_path), *argv]))

    run.suite_path = suite_path  # type: ignore[attr-defined]
    yield run
    get_settings.cache_clear()
    sys.modules.pop("promote", None)


def _report_wrong_answer(trace_id: str = "t1", correction: str = "It is 30 days.") -> str:
    """What POST /v1/feedback does for an INCORRECT verdict."""
    case = Case(
        case_id=casebook.feedback_case_id(trace_id),
        owner_id="demo",
        trace_id=trace_id,
        question="How many days do customers have to request a refund?",
        outcome=CaseOutcome.ANSWERED,
        diagnosis=CaseDiagnosis.USER_REPORTED_INCORRECT,
        created_at="2026-09-20T10:00:00Z",
        correction=correction,
    )
    casebook.record_feedback(
        owner_id="demo", trace_id=trace_id, verdict=FeedbackVerdict.INCORRECT,
        day="2026-09-20", case=case,
    )
    return case.case_id


def test_the_whole_loop_from_reader_report_to_enforced_regression(
    cli: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """A reader says the answer was wrong, a person reviews and promotes it,
    and it lands in the suite as a pending regression carrying the
    correction — then leaves the review list, so nobody promotes it twice."""
    case_id = _report_wrong_answer(correction="It is 30 days.")

    assert cli("list") == 0
    listed = capsys.readouterr().out
    assert case_id in listed
    assert "reader says: It is 30 days." in listed

    assert cli("promote", case_id) == 0
    suite = json.loads(cli.suite_path.read_text(encoding="utf-8"))
    [item] = suite["active"]
    assert item["id"] == "R001"
    assert item["status"] == "pending"
    assert item["reference_answer"] == "It is 30 days."
    assert item["source"]["case_id"] == case_id

    capsys.readouterr()
    cli("list")
    assert case_id not in capsys.readouterr().out

    assert cli("lock", "R001") == 0
    assert json.loads(cli.suite_path.read_text(encoding="utf-8"))["active"][0]["status"] == (
        "enforced"
    )


def test_a_refused_promotion_exits_nonzero_and_writes_nothing(
    cli: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    case_id = _report_wrong_answer()

    assert cli("promote", case_id, "--expect", "refuse") == 1

    assert "--note" in capsys.readouterr().err
    assert not cli.suite_path.exists()


def test_promoting_an_unknown_case_explains_where_to_look(
    cli: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli("promote", "nope") == 1
    assert "Run `list`" in capsys.readouterr().err


def test_retire_through_the_cli_keeps_the_record(cli: Any) -> None:
    case_id = _report_wrong_answer()
    cli("promote", case_id)

    assert cli("retire", "R001", "--reason", "Policy changed to 60 days.") == 0

    suite = json.loads(cli.suite_path.read_text(encoding="utf-8"))
    assert suite["active"] == []
    assert suite["retired"][0]["reason"] == "Policy changed to 60 days."
