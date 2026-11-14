"""POST /v1/query and POST /v1/query/stream.

Role in architecture: thin HTTP wrappers over api/query_pipeline.py, which
owns the guardrail order, the response contract and the bookkeeping. The two
routes differ only in transport:

  /v1/query          one JSON body after the graph completes
  /v1/query/stream   Server-Sent Events: one `stage` event per completed node,
                     then one `result` event carrying the exact body
                     /v1/query would have returned

The stream carries stage TELEMETRY, never answer text, until the very last
event. A draft is not an answer until the critic and the deterministic
grounding gate have passed it, and a client that rendered draft tokens as
they arrived would show readers precisely the text the system might be about
to refuse (ADR 0005, ADR 0010).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.api import query_pipeline as pipeline
from app.guardrails.auth import resolve_user
from app.models.schemas import QueryRequest, QueryResponse, RefusalResponse

router = APIRouter()


def _get_graph() -> Any:
    """Seam for tests: replaced with a stub graph in the HTTP suites."""
    return pipeline.default_graph()


@router.post("/query", response_model=None)
def query(
    req: QueryRequest,
    user: dict[str, Any] = Depends(resolve_user),
) -> QueryResponse | RefusalResponse:
    prepared = pipeline.prepare(req, user)
    cached = pipeline.from_cache(prepared)
    if cached is not None:
        return cached
    result = pipeline.run(prepared, _get_graph())
    return pipeline.finalize(prepared, result)


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


def _body(response: QueryResponse | RefusalResponse) -> dict[str, Any]:
    return response.model_dump(mode="json", by_alias=True)


@router.post("/query/stream")
def query_stream(
    req: QueryRequest,
    user: dict[str, Any] = Depends(resolve_user),
) -> StreamingResponse:
    # Guardrails run BEFORE the stream opens, so a blocked request still gets
    # its proper HTTP status (401/413/422/429/404/409) rather than a 200 whose
    # first event is an error.
    prepared = pipeline.prepare(req, user)
    graph = _get_graph()

    def events() -> Iterator[str]:
        yield _sse("start", {"trace_id": prepared.trace.trace_id})
        cached = pipeline.from_cache(prepared)
        if cached is not None:
            yield _sse("stage", pipeline.stage_event("cache", prepared.trace, prepared.state))
            yield _sse("result", _body(cached))
            return
        final: dict[str, Any] = {}
        try:
            for node, state in pipeline.iterate(graph, prepared.state):
                if node is None:
                    final = state
                else:
                    yield _sse("stage", pipeline.stage_event(node, prepared.trace, state))
        except Exception as exc:  # noqa: BLE001 — reported as an event, see below
            failure: HTTPException = pipeline.upstream_failure(prepared, exc)
            # The status line has already been sent; the error travels as an
            # event the client renders exactly as it would a 503.
            yield _sse("error", {"status": failure.status_code, "detail": failure.detail})
            return
        yield _sse("result", _body(pipeline.finalize(prepared, final)))

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
