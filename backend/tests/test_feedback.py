"""Reader feedback: the one failure signal the system cannot generate itself.

Every other case in the casebook is the graph noticing its own failure. These
tests cover the class it cannot notice — an answer that passed every gate and
was still wrong — and the properties that make a report of it trustworthy
enough to act on and, later, to promote into a regression test:

  * a verdict is recorded once per reader per answer, whatever they click
  * a partial failure never leaves the reader told "already rated" for a
    verdict that was never stored
  * the question on a case is the one actually asked, not one the client sent
  * nobody can rate, or learn of, another tenant's answer
"""

from __future__ import annotations

from typing import Any

import boto3
import pytest
from app.config import get_settings
from app.learning import casebook
from app.models.schemas import (
    Case,
    CaseDiagnosis,
    CaseOutcome,
    FeedbackVerdict,
    RefusalReason,
)
from fastapi.testclient import TestClient
from moto import mock_aws

DEMO = {"x-api-key": "demo-local"}
OTHER_TENANT = {"x-api-key": "admin-local"}

ASKED_AT = "2026-09-20T10:00:00Z"
DAY = ASKED_AT[:10]


# ------------------------------------------------------------------ fixtures


@pytest.fixture()
def store(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("INDEX_DIR", str(tmp_path))
    monkeypatch.setenv("LOCAL_MODE", "true")
    get_settings.cache_clear()
    casebook.reset_local()
    yield tmp_path
    get_settings.cache_clear()


@pytest.fixture()
def ddb(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("LOCAL_MODE", "false")
    get_settings.cache_clear()
    with mock_aws():
        settings = get_settings()
        boto3.resource("dynamodb", region_name=settings.aws_region).create_table(
            TableName=settings.ddb_table_documents,
            KeySchema=[
                {"AttributeName": "pk", "KeyType": "HASH"},
                {"AttributeName": "sk", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "pk", "AttributeType": "S"},
                {"AttributeName": "sk", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )
        yield
    get_settings.cache_clear()


@pytest.fixture()
def client(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("INDEX_DIR", str(tmp_path))
    monkeypatch.setenv("LOCAL_MODE", "true")
    get_settings.cache_clear()

    from app.agents import retriever
    from app.main import create_app
    from app.observability import tracing

    retriever.reset_cache()
    casebook.reset_local()
    tracing._local_traces.clear()
    with TestClient(create_app()) as c:
        yield c
    get_settings.cache_clear()
    retriever.reset_cache()
    tracing._local_traces.clear()


def _case(trace_id: str = "t1", *, correction: str | None = None) -> Case:
    return Case(
        case_id=casebook.feedback_case_id(trace_id),
        owner_id="u",
        trace_id=trace_id,
        question="How many days do customers have to request a refund?",
        outcome=CaseOutcome.ANSWERED,
        diagnosis=CaseDiagnosis.USER_REPORTED_INCORRECT,
        created_at=ASKED_AT,
        correction=correction,
    )


def _record(verdict: FeedbackVerdict, trace_id: str = "t1", case: Case | None = None) -> bool:
    return casebook.record_feedback(
        owner_id="u", trace_id=trace_id, verdict=verdict, day=DAY, case=case
    )


def _stub_graph(monkeypatch: pytest.MonkeyPatch, **outcome: Any) -> None:
    from app.api import routes_query

    class _G:
        def invoke(self, state: dict) -> dict:
            state.update(outcome)
            return state

    monkeypatch.setattr(routes_query, "_get_graph", lambda: _G())


def _answers(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_graph(
        monkeypatch,
        status="answered",
        answer="Customers have 14 days to request a refund.",
        retrieved=[],
        citations=[],
    )


def _ask(client: Any, question: str, headers: dict[str, str] = DEMO) -> str:
    body = client.post("/v1/query", json={"query": question}, headers=headers).json()
    return str(body["trace_id"])


# ------------------------------------------------------ casebook, local mode


def test_a_wrong_answer_opens_a_case_carrying_the_correction(store: Any) -> None:
    """The correction is what turns a complaint into a future test: it says
    what passing looks like, which a refusal never can."""
    assert _record(
        FeedbackVerdict.INCORRECT, case=_case(correction="30 days, not 14.")
    ) is True

    [case] = casebook.list_for_owner("u")
    assert case.diagnosis == CaseDiagnosis.USER_REPORTED_INCORRECT
    assert case.correction == "30 days, not 14."


def test_a_helpful_verdict_is_counted_but_opens_no_case(store: Any) -> None:
    """A good answer is not a failure. Filing it would bury the cases that
    matter; counting it is what gives 'marked wrong' a denominator."""
    assert _record(FeedbackVerdict.HELPFUL) is True

    assert casebook.list_for_owner("u") == []
    assert casebook.feedback_totals_for_owner("u", since_date=DAY) == (1, 1, 0)


def test_a_reader_rates_an_answer_once_whatever_they_click(store: Any) -> None:
    """Helpful then wrong must not count as two ratings. Deduping on the case
    alone would miss this — HELPFUL opens no case to collide with."""
    assert _record(FeedbackVerdict.HELPFUL) is True
    assert _record(FeedbackVerdict.INCORRECT, case=_case()) is False
    assert _record(FeedbackVerdict.INCORRECT, case=_case()) is False

    assert casebook.list_for_owner("u") == []  # the first verdict stands
    assert casebook.feedback_totals_for_owner("u", since_date=DAY) == (1, 1, 0)


def test_a_failed_write_leaves_no_claim_so_the_retry_is_recorded(
    store: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ordering bug this design exists to prevent. If the claim landed
    first and the case write then failed, the reader's retry would be told
    'already rated' for a report that does not exist — silently lost."""
    real = casebook._local_write_all
    calls = {"n": 0}

    def flaky(rows: list[dict[str, Any]]) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("disk full")
        real(rows)

    monkeypatch.setattr(casebook, "_local_write_all", flaky)

    with pytest.raises(OSError):
        _record(FeedbackVerdict.INCORRECT, case=_case(correction="30 days"))

    assert _record(FeedbackVerdict.INCORRECT, case=_case(correction="30 days")) is True
    [case] = casebook.list_for_owner("u")
    assert case.correction == "30 days"
    assert casebook.feedback_totals_for_owner("u", since_date=DAY) == (1, 0, 1)


def test_different_answers_are_rated_independently(store: Any) -> None:
    assert _record(FeedbackVerdict.INCORRECT, "t1", _case("t1")) is True
    assert _record(FeedbackVerdict.INCORRECT, "t2", _case("t2")) is True

    assert len(casebook.list_for_owner("u")) == 2


def test_reported_answers_reach_the_owner_work_list(store: Any) -> None:
    """A wrong answer is at least as urgent as an unanswerable question: a
    refusal wastes a reader's time, a wrong answer gets acted on."""
    from datetime import UTC, datetime

    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    for i, q in enumerate([
        "How many days do customers have to request a refund?",
        "How many days is the refund window?",
    ]):
        case = _case(f"t{i}").model_copy(update={"question": q, "created_at": now})
        casebook.record_feedback(
            owner_id="u", trace_id=f"t{i}", verdict=FeedbackVerdict.INCORRECT,
            day=now[:10], case=case,
        )
    casebook.record_feedback(
        owner_id="u", trace_id="t9", verdict=FeedbackVerdict.HELPFUL, day=now[:10], case=None
    )

    report = casebook.knowledge_gaps("u")

    [gap] = report.gaps
    assert gap.diagnosis == CaseDiagnosis.USER_REPORTED_INCORRECT
    assert gap.question_count == 2
    assert "superseded" in gap.recommended_action
    assert (report.answers_rated, report.marked_helpful, report.marked_wrong) == (3, 1, 2)


# ------------------------------------------------------------- DynamoDB path


def test_dynamodb_records_a_verdict_once_atomically(ddb: Any) -> None:
    """Two concurrent clicks on two Lambda instances must not both win. The
    claim is a condition inside the same transaction as the case and the
    counters, so the loser writes nothing at all."""
    assert _record(FeedbackVerdict.INCORRECT, case=_case(correction="30 days")) is True
    assert _record(FeedbackVerdict.HELPFUL) is False

    [case] = casebook.list_for_owner("u")
    assert case.correction == "30 days"
    # The losing HELPFUL moved no counter: the transaction was all-or-nothing.
    assert casebook.feedback_totals_for_owner("u", since_date=DAY) == (1, 0, 1)


def test_dynamodb_a_helpful_verdict_opens_no_case(ddb: Any) -> None:
    assert _record(FeedbackVerdict.HELPFUL) is True

    assert casebook.list_for_owner("u") == []
    assert casebook.feedback_totals_for_owner("u", since_date=DAY) == (1, 1, 0)


def test_dynamodb_feedback_counters_share_a_row_with_answer_counters(ddb: Any) -> None:
    """One read per report: the answer rate and the feedback counts come from
    the same daily row, so neither query can see a day the other missed."""
    casebook.bump_totals("u", CaseOutcome.ANSWERED)
    today = casebook._today()
    casebook.record_feedback(
        owner_id="u", trace_id="t1", verdict=FeedbackVerdict.HELPFUL, day=today, case=None
    )

    assert casebook.totals_for_owner("u", since_date=today) == (1, 0)
    assert casebook.feedback_totals_for_owner("u", since_date=today) == (1, 1, 0)


# ------------------------------------------------------------------ the API


def test_rating_a_wrong_answer_reaches_the_owner_report(
    client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE end-to-end case: ask, get a confident wrong answer, say so, and the
    person who owns the refund policy finds out."""
    _answers(monkeypatch)
    trace_id = _ask(client, "How many days do customers have to request a refund?")

    resp = client.post(
        "/v1/feedback",
        json={"trace_id": trace_id, "verdict": "INCORRECT", "correction": "It is 30 days."},
        headers=DEMO,
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["recorded"] is True
    assert body["case_id"] == casebook.feedback_case_id(trace_id)

    report = client.get("/v1/insights/knowledge-gaps", headers=DEMO).json()
    assert report["marked_wrong"] == 1
    [gap] = report["gaps"]
    assert gap["diagnosis"] == "USER_REPORTED_INCORRECT"


def test_rating_twice_is_acknowledged_but_not_counted(
    client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _answers(monkeypatch)
    trace_id = _ask(client, "How many days do customers have to request a refund?")

    first = client.post(
        "/v1/feedback", json={"trace_id": trace_id, "verdict": "HELPFUL"}, headers=DEMO
    ).json()
    second = client.post(
        "/v1/feedback", json={"trace_id": trace_id, "verdict": "INCORRECT"}, headers=DEMO
    ).json()

    assert first["recorded"] is True
    assert second["recorded"] is False
    assert second["case_id"] is None
    report = client.get("/v1/insights/knowledge-gaps", headers=DEMO).json()
    assert (report["answers_rated"], report["marked_helpful"], report["marked_wrong"]) == (1, 1, 0)


def test_another_tenants_answer_is_not_found_rather_than_forbidden(
    client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """403 would confirm the trace exists. Trace ids are the handle on another
    tenant's questions, so their existence is itself not ours to reveal."""
    _answers(monkeypatch)
    trace_id = _ask(client, "What is our acquisition target?", headers=OTHER_TENANT)

    resp = client.post(
        "/v1/feedback", json={"trace_id": trace_id, "verdict": "INCORRECT"}, headers=DEMO
    )

    assert resp.status_code == 404
    other = client.get("/v1/insights/knowledge-gaps", headers=OTHER_TENANT).json()
    assert other["answers_rated"] == 0


def test_an_unknown_answer_is_not_found(client: Any) -> None:
    resp = client.post(
        "/v1/feedback", json={"trace_id": "no-such-trace", "verdict": "HELPFUL"}, headers=DEMO
    )
    assert resp.status_code == 404


def test_the_question_on_the_case_is_the_one_actually_asked(
    client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Corrections feed the regression suite. If the client could supply the
    question, it could attach any question to a real answer and plant a
    'known correct' answer in the tests the system is held to."""
    _answers(monkeypatch)
    trace_id = _ask(client, "How many days do customers have to request a refund?")

    client.post(
        "/v1/feedback",
        json={
            "trace_id": trace_id,
            "verdict": "INCORRECT",
            "correction": "Unlimited refunds forever.",
            "question": "Can I get a refund on anything at any time?",
        },
        headers=DEMO,
    )

    [case] = casebook.list_for_owner("demo")
    assert case.question == "How many days do customers have to request a refund?"


def test_a_refusal_can_be_reported_as_a_false_refusal(
    client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """'You refused, but the handbook covers this' is a false refusal reported
    by the one person positioned to notice. Its correction is exactly the test
    that stops the system refusing that question again."""
    _stub_graph(
        monkeypatch,
        status="refused",
        answer="INSUFFICIENT_CONTEXT",
        refusal_reason=RefusalReason.INSUFFICIENT_EVIDENCE,
        retrieved=[],
    )
    trace_id = _ask(client, "What is the refund window?")

    resp = client.post(
        "/v1/feedback",
        json={
            "trace_id": trace_id,
            "verdict": "INCORRECT",
            "correction": "The refund policy says 30 days.",
        },
        headers=DEMO,
    )

    assert resp.json()["recorded"] is True
    cases = [
        c for c in casebook.list_for_owner("demo")
        if c.diagnosis == CaseDiagnosis.USER_REPORTED_INCORRECT
    ]
    [case] = cases
    assert case.outcome == CaseOutcome.REFUSED
    assert case.correction == "The refund policy says 30 days."


def test_the_correction_is_scrubbed_before_it_is_stored(
    client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Readers paste whatever is in front of them. A correction is user text
    and gets the same PII treatment as a query before anything persists it."""
    _answers(monkeypatch)
    trace_id = _ask(client, "Who approves refunds?")

    client.post(
        "/v1/feedback",
        json={
            "trace_id": trace_id,
            "verdict": "INCORRECT",
            "correction": "Email jane.doe@example.com, she approves them.",
        },
        headers=DEMO,
    )

    [case] = casebook.list_for_owner("demo")
    assert case.correction is not None
    assert "jane.doe@example.com" not in case.correction
    assert "[PII:" in case.correction


def test_a_request_that_errored_has_nothing_to_rate(client: Any) -> None:
    from app.observability.tracing import put_trace

    put_trace({
        "trace_id": "errored", "user_id": "demo", "query_redacted": "q",
        "final_status": "error", "steps": [],
    })

    resp = client.post(
        "/v1/feedback", json={"trace_id": "errored", "verdict": "INCORRECT"}, headers=DEMO
    )
    assert resp.status_code == 409


def test_an_oversized_correction_is_rejected(
    client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A correction is 'what should it have said', not a support ticket."""
    _answers(monkeypatch)
    trace_id = _ask(client, "How long do refunds take?")

    resp = client.post(
        "/v1/feedback",
        json={"trace_id": trace_id, "verdict": "INCORRECT", "correction": "x" * 2001},
        headers=DEMO,
    )
    assert resp.status_code == 400


def test_feedback_requires_authentication(client: Any) -> None:
    resp = client.post("/v1/feedback", json={"trace_id": "t", "verdict": "HELPFUL"})
    assert resp.status_code == 400  # missing header -> validation problem, as elsewhere
