"""Central typed configuration.

Role in architecture: the ONLY module that reads environment variables.
Every other module calls get_settings(); nothing else touches os.environ.
Validated at import/startup so a misconfigured deploy fails fast and loudly.

Two switches shape everything else:

  STORAGE_BACKEND = local | aws
      local: JSON files under INDEX_DIR for the registry, casebook, traces
             and index; API keys from an in-process fixture. Zero cloud
             dependencies — this is what `docker compose up` runs.
      aws:   DynamoDB for registry/casebook/traces/quotas/locks, S3 for the
             index, SSM for the provider key. Free tier throughout.

  LLM_PROVIDER = groq | openai | azure | anthropic | local
      Picks the chat-completion adapter in app/llm. Every adapter shares one
      retry/backoff/circuit-breaker/deadline core, so switching providers
      changes an HTTP shape, never the system's behaviour under failure.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

StorageBackend = Literal["local", "aws"]
LLMProvider = Literal["groq", "openai", "azure", "anthropic", "local"]


class Settings(BaseSettings):
    """All Sentinel configuration, typed and validated by pydantic-settings."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- deployment shape ---
    # Empty means "derive from the legacy LOCAL_MODE flag" so an existing .env
    # or the Lambda image's baked environment keeps working unchanged.
    storage_backend: StorageBackend | str = ""
    # True on laptops: auth/quota/traces use in-memory fixtures instead of DynamoDB.
    # Derived from storage_backend when that is set; kept as a field because
    # nearly every module reads it and because tests set LOCAL_MODE directly.
    local_mode: bool = True
    use_s3_index: bool = False
    # Enables the chaos-demo endpoint and its fault injection. Refused unless
    # the storage backend is local: fault injection never ships to production.
    demo_mode: bool = False

    # --- LLM provider selection ---
    llm_provider: LLMProvider = "groq"

    # Groq. Locally: set GROQ_API_KEY in .env. In AWS: leave it unset and set
    # GROQ_API_KEY_SSM_PARAM to the name of an SSM SecureString instead, so the
    # secret never appears in the function's configuration (see get_settings).
    groq_api_key: str = ""
    groq_api_key_ssm_param: str = ""
    groq_model_simple: str = "openai/gpt-oss-20b"
    groq_model_complex: str = "openai/gpt-oss-120b"

    # OpenAI, and Azure OpenAI (same wire format, different URL and auth header).
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model_simple: str = "gpt-4o-mini"
    openai_model_complex: str = "gpt-4o"
    azure_openai_endpoint: str = ""  # https://<resource>.openai.azure.com
    azure_openai_api_version: str = "2024-10-21"
    # Azure routes by DEPLOYMENT name, not model name; these are deployments.
    azure_deployment_simple: str = ""
    azure_deployment_complex: str = ""

    # Anthropic Messages API. Model ids are deliberately not defaulted here:
    # they change with releases, and the operator sets the ones their account
    # has access to.
    anthropic_api_key: str = ""
    anthropic_base_url: str = "https://api.anthropic.com"
    anthropic_model_simple: str = ""
    anthropic_model_complex: str = ""

    # Local / air-gapped: Ollama or vLLM, both speak the OpenAI chat format.
    local_llm_base_url: str = "http://localhost:11434/v1"
    local_llm_api_key: str = "unused"
    local_model_simple: str = "llama3.1:8b"
    local_model_complex: str = "llama3.1:70b"

    # Per-1M-token prices as "provider/model=in:out" pairs, comma separated,
    # layered over the built-in table in app/llm/pricing.py. Lets an operator
    # keep cost telemetry honest without a code change when a price moves.
    llm_pricing_overrides: str = ""

    # --- RAG / index ---
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    # "torch" locally (full sentence-transformers); "onnx" in Lambda and in the
    # compose image (quantized MiniLM + onnxruntime — same vectors, ~60 MB
    # instead of ~1.5 GB)
    embed_backend: str = "torch"
    onnx_model_dir: Path = Path("models/onnx")
    # Threads per ONNX inference. A query embeds one short sentence and many
    # requests run at once, so parallelism belongs across requests, not inside
    # one: the default (a thread per host core) oversubscribes a CPU-limited
    # container or a 1-vCPU Lambda and burns cycles spin-waiting.
    embed_threads: int = 1
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
    parser_version: str = "pdf-v3"

    # --- agent hard caps ---
    token_budget: int = 10_000
    deadline_seconds: float = 45.0
    max_attempts: int = 2
    # A rewrite-or-escalate round costs a full retriever+synthesizer+critic
    # pass (~1.5-2k tokens at top_k=5). Below this reserve, a repair attempt
    # would start, spend tokens, and still fail budget partway through —
    # worse than refusing immediately. Route straight to refusal instead.
    min_repair_token_reserve: int = 1500
    # Deterministic model downgrade (agents/budget.pick_model). When less than
    # this much wall clock or token budget remains, the cheap model is used
    # even where the router or escalation chose the large one. A rule, not a
    # model call: ADR 0002 retired the LLM router for measured reasons, and a
    # budget decision made by an LLM would reintroduce that cost on the
    # exact requests that can least afford it.
    downgrade_deadline_seconds: float = 8.0
    downgrade_token_threshold: int = 2500

    # Where the regression suite lives. The control plane writes it only in
    # local storage, where the server runs next to the repository.
    regressions_path: Path = Path("evals/regressions.json")
    # Extra browser origins allowed by CORS, comma separated. Local mode always
    # allows the Vite dev server and the compose web container.
    cors_origins: str = ""

    # --- verified-answer cache ---
    # An answer that passed the critic and the grounding gate for THIS exact
    # question against THIS index version is deterministic evidence-backed
    # output; serving it again costs nothing and cannot be less grounded than
    # the first time. Keyed on (tenant, index version, normalised question),
    # so a corpus change invalidates everything by construction. 0 disables.
    answer_cache_ttl_seconds: int = 3600
    answer_cache_max_entries: int = 512

    # --- AWS ---
    aws_region: str = "ap-south-1"
    s3_bucket_docs: str = ""
    ddb_table_users: str = "sentinel-users"
    ddb_table_quotas: str = "sentinel-quotas"
    ddb_table_traces: str = "sentinel-traces"
    ddb_table_documents: str = "sentinel-documents"

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
    # How often a warm process re-checks whether a NEWER index version has been
    # published elsewhere. One small pointer read, at most once per window and
    # only when a query actually arrives — never a background poller.
    # 0 disables the check (tests, and single-process local runs).
    index_refresh_seconds: int = 60
    # Attempts per LLM call before giving up. Low by default because an
    # interactive request has a person waiting. The eval harness raises it.
    llm_max_retries: int = 3

    # --- derived views -------------------------------------------------------

    @model_validator(mode="after")
    def _reconcile_backend(self) -> "Settings":
        """One source of truth for the deployment shape.

        STORAGE_BACKEND wins when set. When it is not, the older LOCAL_MODE /
        USE_S3_INDEX pair is honoured and STORAGE_BACKEND is derived from it,
        so every existing environment file and the Lambda image keep working.

        The S3 bucket is required up front only when STORAGE_BACKEND=aws is
        chosen explicitly. The legacy LOCAL_MODE=false shape keeps its old
        contract (the bucket is checked where S3 is first used), because
        DynamoDB-only tooling and tests rely on it.
        """
        explicit = bool(self.storage_backend)
        if explicit:
            if self.storage_backend not in ("local", "aws"):
                raise ValueError(
                    f"STORAGE_BACKEND must be 'local' or 'aws', got {self.storage_backend!r}"
                )
            self.local_mode = self.storage_backend == "local"
            self.use_s3_index = self.storage_backend == "aws"
        else:
            self.storage_backend = "local" if self.local_mode else "aws"
        if explicit and self.storage_backend == "aws" and not self.s3_bucket_docs:
            raise ValueError("STORAGE_BACKEND=aws requires S3_BUCKET_DOCS")
        if self.demo_mode and self.storage_backend != "local":
            raise ValueError("DEMO_MODE is only permitted with STORAGE_BACKEND=local")
        return self

    @property
    def llm_configured(self) -> bool:
        """Whether the active provider has the credentials it needs.

        Not a liveness check: a configured key can still be revoked. It is
        what lets the chaos demo say "simulated" instead of failing with a
        401 on a fresh clone that has no key yet.
        """
        match self.llm_provider:
            case "groq":
                return bool(self.groq_api_key)
            case "openai" | "azure":
                return bool(self.openai_api_key)
            case "anthropic":
                return bool(self.anthropic_api_key and self.anthropic_model_simple)
            case "local":
                return True
        return False  # pragma: no cover

    @property
    def model_simple(self) -> str:
        """The cheap model for the active provider (synthesis, critic, rewrite)."""
        return self._models()[0]

    @property
    def model_complex(self) -> str:
        """The strong model for the active provider (router escalation)."""
        return self._models()[1]

    def _models(self) -> tuple[str, str]:
        match self.llm_provider:
            case "groq":
                return self.groq_model_simple, self.groq_model_complex
            case "openai":
                return self.openai_model_simple, self.openai_model_complex
            case "azure":
                return self.azure_deployment_simple, self.azure_deployment_complex
            case "anthropic":
                return self.anthropic_model_simple, self.anthropic_model_complex
            case "local":
                return self.local_model_simple, self.local_model_complex
        raise ValueError(f"unknown LLM_PROVIDER {self.llm_provider!r}")  # pragma: no cover


def _fetch_ssm_secret(name: str, region: str) -> str:
    """Read one SecureString from SSM Parameter Store, decrypted.

    Fails loudly rather than returning "": a missing or unreadable secret is a
    broken deploy, and discovering it at the first provider call — as a 401
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
    on every request. Rotating the parameter reaches NEW containers only.

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
