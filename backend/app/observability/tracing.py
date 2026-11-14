"""Per-request trace: what every node did, what it cost, how long it took.

Role in architecture: nodes only ever call record_step() and add_cost(); the
persistence backend (in-memory locally, DynamoDB in AWS) is swapped without
touching any node. The stored shape is TraceRecord in schemas.py, and it is
what the trace inspector, the telemetry aggregation and the eval harness all
read, so a field added here reaches all three.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TraceRecorder:
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = "local"
    tenant_id: str = ""
    query_redacted: str = ""
    model_path: list[str] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    critic_scores: list[dict[str, float]] = field(default_factory=list)
    repair_count: int = 0
    est_cost_usd: float = 0.0
    provider: str = ""
    cache_hit: bool = False
    _t0: float = field(default_factory=time.monotonic)

    def record_step(
        self, name: str, duration_ms: int, tokens_in: int = 0, tokens_out: int = 0, **meta: object
    ) -> None:
        self.steps.append(
            {
                "name": name,
                "started_ms": int((time.monotonic() - self._t0) * 1000) - duration_ms,
                "duration_ms": duration_ms,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "meta": {k: str(v) for k, v in meta.items()},
            }
        )

    def add_cost(self, usd: float, provider: str) -> None:
        self.est_cost_usd = round(self.est_cost_usd + usd, 8)
        self.provider = provider

    def total_tokens(self) -> int:
        return sum(s["tokens_in"] + s["tokens_out"] for s in self.steps)

    def elapsed_ms(self) -> int:
        return int((time.monotonic() - self._t0) * 1000)

    def to_dict(self, final_status: str) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "query_redacted": self.query_redacted,
            "model_path": self.model_path,
            "provider": self.provider,
            "steps": self.steps,
            "critic_scores": self.critic_scores,
            "repair_count": self.repair_count,
            "final_status": final_status,
            "total_tokens": self.total_tokens(),
            "est_cost_usd": self.est_cost_usd,
            "cache_hit": self.cache_hit,
            "latency_ms": self.elapsed_ms(),
        }

    def dump(self, final_status: str) -> str:
        return json.dumps(self.to_dict(final_status), indent=2)


# ------------------------------------------------------------ trace store
# Single PutItem at request end (never partial mid-request writes).
# local_mode keeps traces in memory so the API works on a laptop.

_local_traces: dict[str, dict[str, Any]] = {}
_LOCAL_MAX = 5_000  # bounded: a long-running compose instance must not grow forever


def put_trace(record: dict[str, Any]) -> None:
    from app.config import get_settings

    settings = get_settings()
    created = record.get("created_at") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    record = {**record, "created_at": created}
    if settings.local_mode:
        _local_traces[record["trace_id"]] = record
        while len(_local_traces) > _LOCAL_MAX:
            _local_traces.pop(next(iter(_local_traces)))
        return
    import boto3

    record["ttl"] = int(time.time()) + 30 * 86400  # NON-NEGOTIABLE 30-day expiry
    table = boto3.resource("dynamodb", region_name=settings.aws_region).Table(
        settings.ddb_table_traces
    )
    table.put_item(Item=_to_ddb(record))


def get_trace(trace_id: str) -> dict[str, Any] | None:
    from app.config import get_settings

    settings = get_settings()
    if settings.local_mode:
        return _local_traces.get(trace_id)
    import boto3

    table = boto3.resource("dynamodb", region_name=settings.aws_region).Table(
        settings.ddb_table_traces
    )
    item: dict[str, Any] | None = table.get_item(Key={"trace_id": trace_id}).get("Item")
    return item


def list_traces(limit: int = 20, *, user_id: str | None = None) -> list[dict[str, Any]]:
    """Most recent traces, newest first.

    AWS mode queries the user_id-created_at GSI when a user is given, which
    is the access pattern the telemetry tab needs; the admin-wide listing
    stays a bounded scan because there is no partition that means "everyone".
    """
    from app.config import get_settings

    settings = get_settings()
    if settings.local_mode:
        rows = list(_local_traces.values())
        if user_id is not None:
            rows = [r for r in rows if r.get("user_id") == user_id]
        return rows[-limit:][::-1]
    import boto3

    table = boto3.resource("dynamodb", region_name=settings.aws_region).Table(
        settings.ddb_table_traces
    )
    if user_id is not None:
        from boto3.dynamodb.conditions import Key

        resp = table.query(
            IndexName="user_id-created_at-index",
            KeyConditionExpression=Key("user_id").eq(user_id),
            ScanIndexForward=False,
            Limit=limit,
        )
        items: list[dict[str, Any]] = resp.get("Items", [])
        return items
    scanned: list[dict[str, Any]] = table.scan(Limit=limit).get("Items", [])
    return scanned


def reset_local() -> None:
    """Test helper."""
    _local_traces.clear()


def _to_ddb(obj: object) -> object:
    """Recursively convert floats to str for DynamoDB (no float type support)."""
    if isinstance(obj, float):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _to_ddb(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_ddb(v) for v in obj]
    return obj
