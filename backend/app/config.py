"""Central typed configuration.

Role in architecture: the ONLY module that reads environment variables.
Every other module calls get_settings(); nothing else touches os.environ.
Validated at import/startup so a misconfigured deploy fails fast and loudly.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All Sentinel configuration, typed and validated by pydantic-settings."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM (Groq) ---
    # Locally: set GROQ_API_KEY in .env. In AWS: leave it unset and set
    # GROQ_API_KEY_SSM_PARAM to the name of an SSM SecureString instead, so the
    # secret never appears in the function's configuration (see get_settings).
    groq_api_key: str = ""
    groq_api_key_ssm_param: str = ""
    groq_model_simple: str = "openai/gpt-oss-20b"
    groq_model_complex: str = "openai/gpt-oss-120b"

    # --- RAG / index ---
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    # "torch" locally (full sentence-transformers); "onnx" in Lambda (quantized
    # MiniLM + onnxruntime — same vectors, ~60 MB instead of ~1.5 GB)
    embed_backend: str = "torch"
    onnx_model_dir: Path = Path("models/onnx")
    index_dir: Path = Path("index")
    chunk_size_tokens: int = 256  # matches all-MiniLM-L6-v2 max seq length
    chunk_overlap_tokens: int = 48
    top_k: int = 5  # kept low: Groq free-tier caps at 8k tokens/min, a repair loop can burn it fast
    candidates_per_retriever: int = 20
    rrf_k: int = 60
    max_upload_bytes: int = 1024 * 1024  # 1 MB per document; authoritative limit

    # --- PDF ingestion limits (a 1 MB file is NOT a bound on work: a small,
    # highly-compressed PDF can still hold hundreds of pages) ---
    max_pdf_pages: int = 200
    max_chunks_per_document: int = 2_000
    # Below this many extractable characters per page on average, a PDF is
    # treated as image-only/scanned rather than silently indexed as empty.
    min_chars_per_page_for_text_pdf: int = 32
    parser_version: str = "pdf-v2"

    # --- agent hard caps ---
    token_budget: int = 10_000
    deadline_seconds: float = 45.0
    max_attempts: int = 2
    # A rewrite-or-escalate round costs a full retriever+synthesizer+critic
    # pass (~1.5-2k tokens at top_k=5). Below this reserve, a repair attempt
    # would start, spend tokens, and still fail budget partway through —
    # worse than refusing immediately. Route straight to refusal instead.
    min_repair_token_reserve: int = 1500

    # --- AWS (used from P4 onward) ---
    aws_region: str = "ap-south-1"
    s3_bucket_docs: str = ""
    ddb_table_users: str = "sentinel-users"
    ddb_table_quotas: str = "sentinel-quotas"
    ddb_table_traces: str = "sentinel-traces"
    ddb_table_documents: str = "sentinel-documents"
    use_s3_index: bool = False

    # --- index publication lock (rag/index_lock.py) ---
    # Sized to the Lambda timeout, not to typical work: a lease shorter than
    # the critical section lets a slow-but-healthy writer have its lock stolen
    # mid-publish, which is the concurrent-writer bug the lock exists to stop.
    index_lock_lease_seconds: int = 120
    # How long a writer waits for the lock before giving up. Kept under the
    # function timeout so the caller fails with a clear error rather than being
    # killed mid-wait.
    index_lock_wait_seconds: int = 60

    # An upload registered this long ago that never delivered its bytes is
    # treated as abandoned. Comfortably beyond the 900s presigned-URL expiry,
    # so a slow-but-real upload is never mistaken for a dead one.
    upload_abandon_seconds: int = 1800
    # True on laptops: auth/quota/traces use in-memory fixtures instead of DynamoDB
    local_mode: bool = True


def _fetch_ssm_secret(name: str, region: str) -> str:
    """Read one SecureString from SSM Parameter Store, decrypted.

    Fails loudly rather than returning "": a missing or unreadable secret is a
    broken deploy, and discovering it at the first Groq call — as a 401
    surfaced to a user mid-request — is far worse than failing at startup.
    The error names the parameter but never includes its value.
    """
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError

    try:
        resp = boto3.client("ssm", region_name=region).get_parameter(
            Name=name, WithDecryption=True
        )
    except (ClientError, BotoCoreError) as exc:
        raise RuntimeError(f"could not read SSM parameter {name!r}: {exc}") from exc
    value: str = resp["Parameter"]["Value"]
    if not value.strip():
        raise RuntimeError(f"SSM parameter {name!r} is empty")
    return value


@lru_cache
def get_settings() -> Settings:
    """Process-wide settings singleton (tests: get_settings.cache_clear()).

    The Groq key is resolved here, once per process. On Lambda that means once
    per container cold start: warm invocations reuse it, so SSM is not called
    on every request. The consequence worth knowing is that rotating the
    parameter reaches NEW containers only — already-warm ones keep the old
    value until Lambda recycles them (or the function is updated, which forces
    fresh containers).

    An explicit GROQ_API_KEY wins over the SSM parameter, so local development
    with a .env file needs no AWS access at all.
    """
    settings = Settings()
    if not settings.groq_api_key and settings.groq_api_key_ssm_param:
        settings = settings.model_copy(
            update={
                "groq_api_key": _fetch_ssm_secret(
                    settings.groq_api_key_ssm_param, settings.aws_region
                )
            }
        )
    return settings
