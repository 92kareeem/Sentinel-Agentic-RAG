"""Input validation beyond schema: byte-size caps.

Role in architecture: Pydantic already enforces field types and the 1..1000
char query length; this adds the byte-level cap a schema can't express
(multi-byte unicode can be 4x the char count) -> 413.
"""

from fastapi import HTTPException

from app.models.schemas import QueryRequest

MAX_QUERY_BYTES = 4096


def validate_query(req: QueryRequest) -> QueryRequest:
    if len(req.query.encode("utf-8")) > MAX_QUERY_BYTES:
        raise HTTPException(status_code=413, detail="query exceeds byte limit")
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query is empty")
    return req
