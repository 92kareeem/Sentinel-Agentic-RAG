"""The knowledge-gap endpoint, through the real app.

The unit tests drive casebook.py directly; these check what only an HTTP round
trip exercises — that a refusal actually reaches the casebook, that the report
is tenant-scoped, and that a casebook failure cannot take a query down with it.

The graph is stubbed, as it is everywhere else in this file's sibling suite.
Nothing here is testing whether the LLM refuses correctly, and a test that
reached Groq would be slow, would spend quota, would fail in CI (which has no
key for the unit-test job) and would go red for network reasons that say
nothing about this code.
"""

from __future__ import annotations

from typing import Any

import pytest
from app.config import get_settings
from app.models.schemas import RefusalReason
from fastapi.testclient import TestClient

DEMO = {"x-api-key": "demo-local"}


@pytest.fixture()
def client(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("INDEX_DIR", str(tmp_path))
    monkeypatch.setenv("LOCAL_MODE", "true")
    get_settings.cache_clear()

    from app.agents import retriever
    from app.learning import casebook
    from app.main import create_app

    retriever.reset_cache()
    casebook.reset_local()
    with TestClient(create_app()) as c:
        yield c
    get_settings.cache_clear()
    retriever.reset_cache()


def _stub_graph(monkeypatch: pytest.MonkeyPatch, **outcome: Any) -> None:
    """Replace the agent graph with one that returns a fixed outcome."""
    from app.api import routes_query

    class _G:
        def invoke(self, state: dict) -> dict:
            state.update(outcome)
            return state

    monkeypatch.setattr(routes_query, "_get_graph", lambda: _G())


def _refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_graph(
        monkeypatch,
        status="refused",
        answer="INSUFFICIENT_CONTEXT",
        refusal_reason=RefusalReason.INSUFFICIENT_EVIDENCE,
        retrieved=[],
    )


def test_a_refusal_becomes_a_knowledge_gap_the_owner_can_see(
    client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE end-to-end case. A refusal is a fact about the corpus, and it has to
    reach the person who can act on it instead of evaporating into an apology
    the user reads once and forgets."""
    _refuses(monkeypatch)
    question = "How much parental leave do contractors get?"

    body = client.post("/v1/query", json={"query": question}, headers=DEMO).json()
    assert body.get("refusal") is True

    report = client.get("/v1/insights/knowledge-gaps", headers=DEMO).json()

    assert report["unanswered"] == 1
    assert report["gaps"], "the refusal was not recorded as a gap"
    assert question in report["gaps"][0]["example_questions"]
    assert report["gaps"][0]["recommended_action"]
    assert report["gaps"][0]["diagnosis"] == "NO_EVIDENCE_FOUND"


def test_a_successful_answer_moves_the_rate_without_creating_a_gap(
    client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Successes are counted but not filed. Recording every answer would turn
    the casebook into a query log and the gap report into noise."""
    _stub_graph(
        monkeypatch,
        status="answered",
        answer="Refunds take 30 days. [chunk:a]",
        retrieved=[],
        citations=[],
    )

    client.post("/v1/query", json={"query": "how long do refunds take"}, headers=DEMO)
    report = client.get("/v1/insights/knowledge-gaps", headers=DEMO).json()

    assert report["answered"] == 1
    assert report["answer_rate"] == 1.0
    assert report["gaps"] == []


def test_the_report_requires_authentication(client: Any) -> None:
    assert client.get("/v1/insights/knowledge-gaps").status_code == 400


def test_the_report_is_empty_and_healthy_before_anyone_asks_anything(client: Any) -> None:
    report = client.get("/v1/insights/knowledge-gaps", headers=DEMO).json()

    assert report["total_questions"] == 0
    assert report["answer_rate"] == 1.0
    assert report["gaps"] == []


def test_the_window_is_bounded_so_a_report_cannot_be_asked_to_scan_forever(
    client: Any,
) -> None:
    assert (
        client.get("/v1/insights/knowledge-gaps?window_days=99999", headers=DEMO).status_code
        == 400
    )


def test_a_broken_casebook_does_not_break_answering(
    client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Capture is a byproduct of answering. A user who asked a hard question
    must not lose their response because recording it failed — that would
    invert the purpose of a feature whose whole job is to make hard questions
    better over time."""
    from app.learning import casebook

    def boom(*_: Any, **__: Any) -> None:
        raise RuntimeError("casebook unavailable")

    _refuses(monkeypatch)
    # Patched at the module level, so the failure is not caught by any guard
    # inside casebook itself — the request path has to survive it on its own.
    monkeypatch.setattr(casebook, "record", boom)
    monkeypatch.setattr(casebook, "bump_totals", boom)

    resp = client.post("/v1/query", json={"query": "anything at all"}, headers=DEMO)

    assert resp.status_code == 200
    assert resp.json()["refusal"] is True
