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
    groq_api_key: str = ""
    groq_model_simple: str = "llama-3.1-8b-instant"
    groq_model_complex: str = "llama-3.3-70b-versatile"

    # --- RAG / index ---
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    # "torch" locally (full sentence-transformers); "onnx" in Lambda (quantized
    # MiniLM + onnxruntime — same vectors, ~60 MB instead of ~1.5 GB)
    embed_backend: str = "torch"
    onnx_model_dir: Path = Path("models/onnx")
    index_dir: Path = Path("index")
    chunk_size_tokens: int = 256  # matches all-MiniLM-L6-v2 max seq length
    chunk_overlap_tokens: int = 48
    top_k: int = 8
    candidates_per_retriever: int = 20
    rrf_k: int = 60
    max_upload_bytes: int = 1024 * 1024  # 1 MB per document (S3-enforced); free-tier safe

    # --- agent hard caps ---
    token_budget: int = 10_000
    deadline_seconds: float = 20.0
    max_attempts: int = 2

    # --- AWS (used from P4 onward) ---
    aws_region: str = "ap-south-1"
    s3_bucket_docs: str = ""
    ddb_table_users: str = "sentinel-users"
    ddb_table_quotas: str = "sentinel-quotas"
    ddb_table_traces: str = "sentinel-traces"
    use_s3_index: bool = False
    # True on laptops: auth/quota/traces use in-memory fixtures instead of DynamoDB
    local_mode: bool = True


@lru_cache
def get_settings() -> Settings:
    """Process-wide settings singleton (tests: get_settings.cache_clear())."""
    return Settings()
