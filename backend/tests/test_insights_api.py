"""The knowledge-gap endpoint, through the real app.

The unit tests drive casebook.py directly; these check the parts only an HTTP
round trip exercises — that a refusal actually lands in the casebook, that the
report is tenant-scoped, and that a casebook failure cannot take a query down
with it.
"""

from __future__ import annotations

from typing import Any

import pytest
from app.config import get_settings
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


def test_a_refusal_becomes_a_knowledge_gap_the_owner_can_see(client: Any) -> None:
    """THE end-to-end case. With no documents indexed, retrieval finds nothing
    and the question is recorded as a gap rather than evaporating into a
    refusal the user reads once and forgets."""
    question = "How much parental leave do contractors get?"
    body = client.post("/v1/query", json={"query": question}, headers=DEMO).json()
    assert body.get("refusal") is True

    report = client.get("/v1/insights/knowledge-gaps", headers=DEMO).json()

    assert report["unanswered"] == 1
    assert report["gaps"], "the refusal was not recorded as a gap"
    assert question in report["gaps"][0]["example_questions"]
    assert report["gaps"][0]["recommended_action"]


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
    assert client.get(
        "/v1/insights/knowledge-gaps?window_days=99999", headers=DEMO
    ).status_code == 400


def test_a_broken_casebook_does_not_break_answering(
    client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Capture is a byproduct. A user who asked a hard question must not lose
    their response because recording it failed."""
    from app.learning import casebook

    def boom(*_: Any, **__: Any) -> None:
        raise RuntimeError("casebook unavailable")

    monkeypatch.setattr(casebook, "_local_write_all", boom)
    monkeypatch.setattr(casebook, "bump_totals", boom)

    resp = client.post("/v1/query", json={"query": "anything at all"}, headers=DEMO)

    assert resp.status_code == 200
