"""The control plane: telemetry, the trace browser, the ratchet, the chaos demo.

Telemetry is pure arithmetic over trace dicts and is tested as such. The
ratchet API is tested for the properties ADR 0008 depends on: promotion is an
admin act, lands as pending, needs a reference for an answer test, writes a
suite the validator accepts, and never deletes. The chaos demo is tested
offline: every scenario must end in the outcome the architecture promises.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from app.api import routes_query
from app.learning import ratchet
from app.models.schemas import RefusalReason
from app.observability import telemetry

from .conftest import chunk, make_client, publish_chunks

DEMO = {"x-api-key": "demo-local"}
ADMIN = {"x-api-key": "admin-local"}
GLOBEX = {"x-api-key": "globex-local"}

NOW = datetime(2026, 11, 14, 12, 0, tzinfo=UTC)


def _trace(status: str, ms: int, minutes_ago: int = 5, **extra: Any) -> dict[str, Any]:
    return {
        "trace_id": f"t{ms}{status}{minutes_ago}",
        "created_at": (NOW - timedelta(minutes=minutes_ago)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "final_status": status,
        "latency_ms": ms,
        "total_tokens": 100,
        "est_cost_usd": 0.001,
        "provider": "groq",
        "repair_count": extra.pop("repairs", 0),
        "cache_hit": extra.pop("cache_hit", False),
        "steps": extra.pop("steps", []),
        **extra,
    }


# ----------------------------------------------------------------- telemetry


def test_telemetry_percentiles_rates_and_cost() -> None:
    traces = [_trace("answered", ms) for ms in (100, 200, 300, 400)]
    traces.append(
        _trace(
            "refused",
            1000,
            steps=[
                {"name": "refusal", "duration_ms": 0, "meta": {"reason": "INSUFFICIENT_EVIDENCE"}},
            ],
        )
    )
    traces.append(_trace("answered", 5, cache_hit=True))
    traces.append(_trace("answered", 50, repairs=1))
    traces.append(_trace("error", 9999))
    r = telemetry.build_report(traces, window_hours=24, now=NOW)

    assert (r.requests, r.answered, r.refused, r.errors) == (8, 6, 1, 1)
    assert r.latency_p50_ms == 200 and r.latency_p95_ms == 1000  # error excluded
    assert r.refusal_rate == pytest.approx(1 / 7, abs=1e-4)
    assert r.cache_hit_rate == pytest.approx(1 / 7, abs=1e-4)
    assert r.repair_rate == pytest.approx(1 / 7, abs=1e-4)
    assert r.est_cost_usd == pytest.approx(0.008)
    assert r.refusal_reasons == {"INSUFFICIENT_EVIDENCE": 1}
    assert len(r.hourly) == 24 and sum(h.requests for h in r.hourly) == 8
    assert r.hourly[-2].requests == 8  # 11:55 belongs to the 11:00 bucket


def test_telemetry_window_excludes_old_traces() -> None:
    r = telemetry.build_report(
        [_trace("answered", 100, minutes_ago=5), _trace("answered", 100, minutes_ago=60 * 30)],
        window_hours=24,
        now=NOW,
    )
    assert r.requests == 1


def test_telemetry_stage_profile_counts_downgrades() -> None:
    steps = [
        {
            "name": "synthesizer",
            "duration_ms": 900,
            "tokens_in": 800,
            "tokens_out": 100,
            "meta": {"downgraded": "True"},
        },
        {"name": "critic", "duration_ms": 300, "tokens_in": 300, "tokens_out": 50, "meta": {}},
    ]
    r = telemetry.build_report([_trace("answered", 1200, steps=steps)], window_hours=1, now=NOW)
    assert [s.name for s in r.stages] == ["synthesizer", "critic"]
    assert r.stages[0].tokens == 900 and r.downgrades == 1


def test_an_empty_window_is_zeros_not_an_error() -> None:
    r = telemetry.build_report([], window_hours=24, now=NOW)
    assert r.requests == 0 and r.refusal_rate == 0.0 and r.latency_p95_ms == 0


# ------------------------------------------------------------ via the API


def _refuse_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    class _G:
        def invoke(self, state: dict) -> dict:
            state.update(
                status="refused",
                answer="INSUFFICIENT_CONTEXT",
                refusal_reason=RefusalReason.INSUFFICIENT_EVIDENCE,
                retrieved=[],
            )
            return state

    monkeypatch.setattr(routes_query, "_get_graph", lambda: _G())


def test_telemetry_and_traces_endpoints_are_scoped(api: Any, monkeypatch) -> None:
    _refuse_graph(monkeypatch)
    api.post("/v1/query", json={"query": "parental leave for contractors?"}, headers=DEMO)
    api.post("/v1/query", json={"query": "what is our revenue?"}, headers=GLOBEX)

    mine = api.get("/v1/insights/telemetry", headers=DEMO).json()
    assert mine["requests"] == 1 and mine["refused"] == 1
    tenant = api.get("/v1/insights/telemetry", headers=ADMIN).json()
    assert tenant["requests"] == 1  # acme admin: sees demo (acme), not globex

    rows = api.get("/v1/insights/traces", headers=DEMO).json()
    assert len(rows) == 1 and rows[0]["final_status"] == "refused"
    assert api.get("/v1/insights/traces", headers=GLOBEX).json()[0]["query_redacted"] == (
        "what is our revenue?"
    )


def test_traces_survive_a_restart(api: Any, monkeypatch) -> None:
    from app.observability import tracing

    _refuse_graph(monkeypatch)
    trace_id = api.post("/v1/query", json={"query": "q one?"}, headers=DEMO).json()["trace_id"]
    tracing.reset_local()  # as if the process restarted
    assert api.get(f"/v1/traces/{trace_id}", headers=DEMO).status_code == 200


# --------------------------------------------------------------- the ratchet


def test_the_full_ratchet_loop_through_the_api(api: Any, monkeypatch, offline_env: Any) -> None:
    _refuse_graph(monkeypatch)
    api.post(
        "/v1/query", json={"query": "How much parental leave do contractors get?"}, headers=DEMO
    )

    view = api.get("/v1/insights/ratchet", headers=ADMIN).json()
    assert view["writable"] is True
    assert len(view["candidates"]) == 1
    case_id = view["candidates"][0]["case_id"]
    assert case_id in view["candidates"][0]["promote_command"]

    # An answer test needs the correct answer; a refusal has no correction.
    no_ref = api.post("/v1/insights/ratchet/promote", json={"case_id": case_id}, headers=ADMIN)
    assert no_ref.status_code == 422

    # Only an admin may write the suite.
    assert (
        api.post(
            "/v1/insights/ratchet/promote",
            json={"case_id": case_id, "reference": "12 weeks."},
            headers=DEMO,
        ).status_code
        == 403
    )

    item = api.post(
        "/v1/insights/ratchet/promote",
        json={"case_id": case_id, "reference": "Contractors get 12 weeks."},
        headers=ADMIN,
    ).json()
    assert item["id"] == "R001" and item["status"] == "pending"

    suite = ratchet.load(offline_env / "regressions.json")
    assert ratchet.validate(suite) == [] and suite.issued == 1

    view = api.get("/v1/insights/ratchet", headers=ADMIN).json()
    assert view["pending"] == 1 and view["candidates"] == []

    # Twice is refused; retiring needs a reason and keeps the record.
    twice = api.post(
        "/v1/insights/ratchet/promote", json={"case_id": case_id, "reference": "x"}, headers=ADMIN
    )
    assert twice.status_code == 422 and "already" in twice.json()["detail"]
    assert (
        api.post(
            "/v1/insights/ratchet/R001/retire", json={"reason": "no"}, headers=ADMIN
        ).status_code
        == 400
    )
    assert (
        api.post(
            "/v1/insights/ratchet/R001/retire",
            json={"reason": "policy changed in 2027"},
            headers=ADMIN,
        ).status_code
        == 200
    )
    suite = ratchet.load(offline_env / "regressions.json")
    assert suite.active == [] and suite.retired[0].id == "R001" and suite.issued == 1


def test_another_tenants_cases_are_not_promotable(api: Any, monkeypatch) -> None:
    _refuse_graph(monkeypatch)
    api.post("/v1/query", json={"query": "globex secret question?"}, headers=GLOBEX)
    assert api.get("/v1/insights/ratchet", headers=ADMIN).json()["candidates"] == []


# --------------------------------------------------------------- chaos demo


@pytest.fixture()
def demo_api(offline_env: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    with make_client(monkeypatch, demo=True) as c:
        yield c


def test_health_reports_the_deployment_shape(demo_api: Any) -> None:
    body = demo_api.get("/healthz").json()
    assert body["demo_mode"] is True and body["storage_backend"] == "local"
    assert body["llm_configured"] is False and body["version"]


def test_the_chaos_route_does_not_exist_outside_demo_mode(api: Any) -> None:
    resp = api.post("/v1/demo/chaos", json={"scenario": "pii_leak"}, headers=DEMO)
    assert resp.status_code == 404


@pytest.mark.parametrize(
    ("scenario", "outcome", "reason"),
    [
        ("prompt_injection", "blocked", None),
        ("pii_leak", "redacted", None),
        ("hallucination_trap", "refused", "UNVERIFIABLE_ANSWER"),
        ("out_of_scope", "refused", "INSUFFICIENT_EVIDENCE"),
        ("budget_exceeded", "degraded", "BUDGET_EXHAUSTED"),
    ],
)
def test_every_chaos_scenario_is_caught_by_the_layer_designed_for_it(
    demo_api: Any, scenario: str, outcome: str, reason: str | None
) -> None:
    publish_chunks([chunk("c1", "Customers may request a refund within 30 days of purchase.")])
    r = demo_api.post("/v1/demo/chaos", json={"scenario": scenario}, headers=DEMO).json()
    assert r["passed"] is True, r
    assert r["outcome"] == outcome
    assert r["refusal_reason"] == reason


def test_the_hallucination_trap_never_leaks_the_fabricated_draft(demo_api: Any) -> None:
    publish_chunks([chunk("c1", "Customers may request a refund within 30 days of purchase.")])
    r = demo_api.post(
        "/v1/demo/chaos", json={"scenario": "hallucination_trap"}, headers=DEMO
    ).json()
    assert r["evidence"]["draft_in_response"] == "false"
    assert r["evidence"]["repairs"] == "2"
    names = [s["name"] for s in r["steps"]]
    assert names.count("grounding_check") == 3 and names[-1] == "refusal"
    trace = demo_api.get(f"/v1/traces/{r['trace_id']}", headers=DEMO).json()
    assert "999" not in str(trace)


def test_the_budget_scenario_shows_the_downgrade(demo_api: Any) -> None:
    publish_chunks([chunk("c1")])
    r = demo_api.post("/v1/demo/chaos", json={"scenario": "budget_exceeded"}, headers=DEMO).json()
    assert r["evidence"]["downgraded"] == "true"
