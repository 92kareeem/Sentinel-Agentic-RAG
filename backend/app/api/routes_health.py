"""GET /healthz — no auth, no quota: load balancers and smoke tests hit this."""

from pathlib import Path

from fastapi import APIRouter

from app.config import get_settings
from app.models.schemas import HealthResponse

router = APIRouter()


@router.get("/healthz")
def healthz() -> HealthResponse:
    """Liveness + what corpus this instance is actually serving.

    Reports the published index version (e.g. "v7") rather than the previous
    mtime of a since-removed faiss.index path, which always read "absent" once
    the index became versioned — and an mtime was never a usable version
    identifier for correlating an answer with the corpus that produced it.
    """
    from app.rag import embeddings, index_store

    settings = get_settings()
    version = index_store.read_pointer(Path(settings.index_dir)) or "absent"

    return HealthResponse(
        status="ok",
        index_version=version,
        model_warm=embeddings._model is not None,
    )
