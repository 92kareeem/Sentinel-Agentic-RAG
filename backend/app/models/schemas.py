"""API contract as code.

Role in architecture: every request/response body in the system is defined here
once. FastAPI validates against these at the edge; frontend/src/types.ts mirrors
them; nothing constructs ad-hoc dicts for API responses.
"""

from enum import StrEnum

from pydantic import BaseModel, Field

# ---------------------------------------------------------------- documents


class DocumentStatus(StrEnum):
    """Explicit document lifecycle. The API never reports INDEXED unless the
    document's chunks are actually queryable in a published index version."""

    UPLOADING = "UPLOADING"  # registered, bytes not yet durably stored
    UPLOADED = "UPLOADED"  # bytes stored, not yet parsed
    PROCESSING = "PROCESSING"  # parse/chunk/embed/index in flight
    INDEXED = "INDEXED"  # queryable
    FAILED = "FAILED"  # terminal; see error_code/error_message
    DELETED = "DELETED"  # tombstoned; chunks purged from the index


class DocumentErrorCode(StrEnum):
    """Machine-readable ingestion failure reasons.

    These are surfaced to the frontend so it can explain WHY a document failed
    rather than showing a generic error — the difference between "this PDF is
    scanned images, try an OCR'd copy" and "something went wrong".
    """

    ENCRYPTED_DOCUMENT = "ENCRYPTED_DOCUMENT"
    UNSUPPORTED_SCANNED_DOCUMENT = "UNSUPPORTED_SCANNED_DOCUMENT"
    CORRUPT_DOCUMENT = "CORRUPT_DOCUMENT"
    EMPTY_DOCUMENT = "EMPTY_DOCUMENT"
    NO_EXTRACTABLE_TEXT = "NO_EXTRACTABLE_TEXT"
    TOO_LARGE = "TOO_LARGE"
    TOO_MANY_PAGES = "TOO_MANY_PAGES"
    UNSUPPORTED_CONTENT_TYPE = "UNSUPPORTED_CONTENT_TYPE"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class DocumentRecord(BaseModel):
    """Canonical document identity and lifecycle state.

    document_id is server-generated at upload and is the ONLY identity used
    downstream. Filenames are user-controlled, non-unique across users and
    across versions, and are therefore metadata only — never identity.
    """

    document_id: str
    owner_id: str
    original_filename: str
    safe_filename: str
    content_type: str
    file_size: int
    checksum_sha256: str
    storage_key: str
    status: DocumentStatus = DocumentStatus.UPLOADING
    created_at: str
    updated_at: str
    page_count: int | None = None
    chunk_count: int = 0
    index_version: str | None = None
    parser_version: str | None = None
    embedding_model: str | None = None
    error_code: DocumentErrorCode | None = None
    error_message: str | None = None


class DocumentSummary(BaseModel):
    """Public projection of DocumentRecord — no storage_key (internal S3 layout
    is not the client's business) and no owner_id (implied by the caller)."""

    document_id: str
    filename: str
    status: DocumentStatus
    file_size: int
    page_count: int | None = None
    chunk_count: int = 0
    created_at: str
    updated_at: str
    error_code: DocumentErrorCode | None = None
    error_message: str | None = None

    @classmethod
    def from_record(cls, record: DocumentRecord) -> "DocumentSummary":
        return cls(
            document_id=record.document_id,
            filename=record.original_filename,
            status=record.status,
            file_size=record.file_size,
            page_count=record.page_count,
            chunk_count=record.chunk_count,
            created_at=record.created_at,
            updated_at=record.updated_at,
            error_code=record.error_code,
            error_message=record.error_message,
        )


# ---------------------------------------------------------------- retrieval


class ChunkType(StrEnum):
    PROSE = "prose"
    TABLE = "table"
    HEADING = "heading"
    LIST = "list"


class Chunk(BaseModel):
    """One retrieval unit; the on-disk record shape of chunks.jsonl."""

    chunk_id: str  # deterministic: "{doc_id}_p{page}_s{section_idx}_c{chunk_idx}"
    doc_id: str  # == DocumentRecord.document_id (server-generated, never a filename)
    owner_id: str = ""  # denormalized for retrieval-time tenant filtering
    section_path: str  # e.g. "Item 7 > Liquidity"
    text: str
    is_table: bool = False
    chunk_type: ChunkType = ChunkType.PROSE
    page_number: int | None = None  # 1-based; None for non-paginated sources
    source_filename: str = ""
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
    page_number: int | None = None
    source_filename: str = ""
    document_id: str = ""


# ---------------------------------------------------------------- /v1/query


class ConversationTurn(BaseModel):
    role: str = Field(min_length=1, max_length=20)
    content: str = Field(min_length=1, max_length=4000)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=8, ge=1, le=20)
    doc_id: str | None = None  # scope retrieval to one uploaded document; None = all
    conversation_history: list[ConversationTurn] = Field(default_factory=list)


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
    upload_url: str  # S3 endpoint the browser POSTs a multipart form to
    fields: dict[str, str]  # signed form fields (policy, signature, key, ...)
    filename: str
    max_bytes: int
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
