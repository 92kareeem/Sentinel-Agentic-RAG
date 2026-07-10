"""API contract as code.

Role in architecture: every request/response body in the system is defined here
once. FastAPI validates against these at the edge; frontend/src/types.ts mirrors
them; nothing constructs ad-hoc dicts for API responses.
"""

from pydantic import BaseModel, Field

# ---------------------------------------------------------------- retrieval


class Chunk(BaseModel):
    """One retrieval unit; the on-disk record shape of chunks.jsonl."""

    chunk_id: str  # deterministic: "{doc_id}_s{section_idx}_c{chunk_idx}"
    doc_id: str
    section_path: str  # e.g. "Item 7 > Liquidity"
    text: str
    is_table: bool = False
    token_count: int
    char_start: int
    char_end: int

    @property
    def embed_text(self) -> str:
        """Text as embedded/indexed: section path prefixed for context."""
        return f"{self.section_path}\n\n{self.text}"


class Citation(BaseModel):
    chunk_id: str
    section_path: str
    quote: str


# ---------------------------------------------------------------- /v1/query


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=8, ge=1, le=20)


class CriticScores(BaseModel):
    faithfulness: float = Field(ge=0.0, le=1.0)
    relevance: float = Field(ge=0.0, le=1.0)


class TokenUsage(BaseModel):
    tokens_in: int = Field(alias="in", default=0)
    tokens_out: int = Field(alias="out", default=0)

    model_config = {"populate_by_name": True}


class QueryResponse(BaseModel):
    trace_id: str
    answer: str
    citations: list[Citation]
    critic: CriticScores
    repair_count: int
    model_used: str
    tokens: TokenUsage
    latency_ms: int


class RefusalResponse(BaseModel):
    trace_id: str
    refusal: bool = True
    reason: str
    best_effort_context: list[str] = []  # chunk_ids we found but couldn't answer from


# ---------------------------------------------------------------- ingestion


class PresignedUploadResponse(BaseModel):
    doc_id: str
    upload_url: str
    expires_in_seconds: int


class IndexJobResponse(BaseModel):
    doc_id: str
    chunks_indexed: int
    index_version: str


# ---------------------------------------------------------------- traces


class TraceStep(BaseModel):
    name: str
    started_ms: int
    duration_ms: int
    tokens_in: int = 0
    tokens_out: int = 0
    meta: dict[str, str] = {}


class TraceRecord(BaseModel):
    trace_id: str
    user_id: str
    created_at: str
    query_redacted: str
    model_path: list[str]
    steps: list[TraceStep]
    critic_scores: list[dict[str, float]]
    repair_count: int
    final_status: str  # "answered" | "refused" | "error"
    total_tokens: int
    est_cost_usd: float
    latency_ms: int


# ---------------------------------------------------------------- misc


class HealthResponse(BaseModel):
    status: str = "ok"
    index_version: str
    model_warm: bool


class Problem(BaseModel):
    """RFC 7807 error shape; every error response uses this, extended with trace_id."""

    type: str = "about:blank"
    title: str
    status: int
    detail: str
    trace_id: str | None = None
