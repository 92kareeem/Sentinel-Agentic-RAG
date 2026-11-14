"""Live telemetry, computed on read from stored traces.

Role in architecture: the control plane's first tab. Everything it reports is
derived from the trace records the graph already writes — no metrics agent,
no time-series database, no scheduled job — so it costs nothing while nobody
is looking, which is the only kind of observability a free-tier deployment
can afford to leave switched on.

Pure functions over a list of dicts, so the arithmetic is unit-tested
without a server, a clock or a database.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

from app.models.schemas import HourBucket, StageStat, TelemetryReport


def _pct(values: list[int], p: float) -> int:
    """Nearest-rank percentile; 0 for an empty list."""
    if not values:
        return 0
    ordered = sorted(values)
    rank = max(1, math.ceil(p * len(ordered)))
    return int(ordered[rank - 1])


def _parse(ts: str) -> datetime | None:
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ"):
        try:
            return datetime.strptime(ts, fmt).replace(tzinfo=UTC)
        except (TypeError, ValueError):
            continue
    return None


def _num(v: Any, default: float = 0.0) -> float:
    """DynamoDB returns Decimals and the trace writer stores floats as strings."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def in_window(
    traces: Iterable[Mapping[str, Any]], window_hours: int, now: datetime | None = None
) -> list[Mapping[str, Any]]:
    now = now or datetime.now(UTC)
    cutoff = now - timedelta(hours=window_hours)
    out = []
    for t in traces:
        ts = _parse(str(t.get("created_at", "")))
        if ts is not None and ts >= cutoff:
            out.append(t)
    return out


def build_report(
    traces: Iterable[Mapping[str, Any]],
    *,
    window_hours: int,
    now: datetime | None = None,
) -> TelemetryReport:
    now = now or datetime.now(UTC)
    rows = in_window(traces, window_hours, now)

    statuses = Counter(str(t.get("final_status", "")) for t in rows)
    answered, refused, errors = statuses["answered"], statuses["refused"], statuses["error"]
    completed = answered + refused
    latencies = [int(_num(t.get("latency_ms"))) for t in rows if t.get("final_status") != "error"]
    tokens = [int(_num(t.get("total_tokens"))) for t in rows]
    cost = sum(_num(t.get("est_cost_usd")) for t in rows)
    cache_hits = sum(1 for t in rows if t.get("cache_hit") in (True, "True", "true"))
    repaired = sum(1 for t in rows if int(_num(t.get("repair_count"))) > 0)

    stage_ms: dict[str, list[int]] = defaultdict(list)
    stage_tokens: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    downgrades = 0
    for t in rows:
        for step in t.get("steps") or []:
            name = str(step.get("name"))
            stage_ms[name].append(int(_num(step.get("duration_ms"))))
            stage_tokens[name] += int(_num(step.get("tokens_in"))) + int(
                _num(step.get("tokens_out"))
            )
            meta = step.get("meta") or {}
            if name == "refusal" and meta.get("reason"):
                reasons[str(meta["reason"])] += 1
            if str(meta.get("downgraded")) == "True":
                downgrades += 1

    order = [
        "cache", "router", "retriever", "synthesizer", "critic",
        "grounding_check", "repair_rewrite", "repair_escalate", "refusal",
    ]
    stages = [
        StageStat(
            name=name,
            calls=len(stage_ms[name]),
            mean_ms=round(sum(stage_ms[name]) / len(stage_ms[name]), 1),
            p95_ms=_pct(stage_ms[name], 0.95),
            tokens=stage_tokens[name],
        )
        for name in sorted(stage_ms, key=lambda n: order.index(n) if n in order else 99)
    ]

    buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for t in rows:
        ts = _parse(str(t.get("created_at", "")))
        if ts is not None:
            buckets[ts.strftime("%Y-%m-%dT%H:00Z")].append(t)
    hourly: list[HourBucket] = []
    start = (now - timedelta(hours=window_hours - 1)).replace(minute=0, second=0, microsecond=0)
    for i in range(window_hours):
        key = (start + timedelta(hours=i)).strftime("%Y-%m-%dT%H:00Z")
        group = buckets.get(key, [])
        hourly.append(
            HourBucket(
                hour=key,
                requests=len(group),
                refused=sum(1 for t in group if t.get("final_status") == "refused"),
                p95_ms=_pct([int(_num(t.get("latency_ms"))) for t in group], 0.95),
            )
        )

    total_tokens = sum(tokens)
    return TelemetryReport(
        generated_at=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        window_hours=window_hours,
        requests=len(rows),
        answered=answered,
        refused=refused,
        errors=errors,
        refusal_rate=round(refused / completed, 4) if completed else 0.0,
        cache_hit_rate=round(cache_hits / completed, 4) if completed else 0.0,
        repair_rate=round(repaired / completed, 4) if completed else 0.0,
        downgrades=downgrades,
        latency_p50_ms=_pct(latencies, 0.50),
        latency_p95_ms=_pct(latencies, 0.95),
        tokens_total=total_tokens,
        tokens_per_request=round(total_tokens / len(rows), 1) if rows else 0.0,
        token_burn_per_hour=round(total_tokens / window_hours, 1),
        est_cost_usd=round(cost, 6),
        est_cost_per_answer_usd=round(cost / answered, 6) if answered else 0.0,
        providers=dict(Counter(str(t.get("provider") or "unknown") for t in rows)),
        refusal_reasons=dict(reasons),
        stages=stages,
        hourly=hourly,
    )
