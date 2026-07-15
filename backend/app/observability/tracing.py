"""Local trace recorder (P2 stub; becomes a DynamoDB writer in P4).

Role in architecture: records per-node timing/tokens in the same shape as
the production TraceRecord (schemas.py). Nodes only ever call
record_step() — P4 swaps the persistence backend without touching any node.
"""

import json
import time
import uuid
from dataclasses import dataclass, field


@dataclass
class TraceRecorder:
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = "local"
    query_redacted: str = ""
    model_path: list[str] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)
    critic_scores: list[dict] = field(default_factory=list)
    repair_count: int = 0
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

    def total_tokens(self) -> int:
        return sum(s["tokens_in"] + s["tokens_out"] for s in self.steps)

    def to_dict(self, final_status: str) -> dict:
        return {
            "trace_id": self.trace_id,
            "user_id": self.user_id,
            "query_redacted": self.query_redacted,
            "model_path": self.model_path,
            "steps": self.steps,
            "critic_scores": self.critic_scores,
            "repair_count": self.repair_count,
            "final_status": final_status,
            "total_tokens": self.total_tokens(),
            "latency_ms": int((time.monotonic() - self._t0) * 1000),
        }

    def dump(self, final_status: str) -> str:
        return json.dumps(self.to_dict(final_status), indent=2)


# ------------------------------------------------------------ trace store
# Single PutItem at request end (never partial mid-request writes).
# local_mode keeps traces in memory so the API works on a laptop.

_local_traces: dict[str, dict] = {}


def put_trace(record: dict) -> None:
    from app.config import get_settings

    settings = get_settings()
    created = record.get("created_at") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    record = {**record, "created_at": created}
    if settings.local_mode:
        _local_traces[record["trace_id"]] = record
        return
    import boto3

    record["ttl"] = int(time.time()) + 30 * 86400  # NON-NEGOTIABLE 30-day expiry
    table = boto3.resource("dynamodb", region_name=settings.aws_region).Table(
        settings.ddb_table_traces
    )
    table.put_item(Item=_to_ddb(record))


def get_trace(trace_id: str) -> dict | None:
    from app.config import get_settings

    settings = get_settings()
    if settings.local_mode:
        return _local_traces.get(trace_id)
    import boto3

    table = boto3.resource("dynamodb", region_name=settings.aws_region).Table(
        settings.ddb_table_traces
    )
    return table.get_item(Key={"trace_id": trace_id}).get("Item")


def list_traces(limit: int = 20) -> list[dict]:
    from app.config import get_settings

    settings = get_settings()
    if settings.local_mode:
        return list(_local_traces.values())[-limit:]
    import boto3

    table = boto3.resource("dynamodb", region_name=settings.aws_region).Table(
        settings.ddb_table_traces
    )
    return table.scan(Limit=limit).get("Items", [])


def _to_ddb(obj: object) -> object:
    """Recursively convert floats to str for DynamoDB (no float type support)."""
    if isinstance(obj, float):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _to_ddb(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_ddb(v) for v in obj]
    return obj
