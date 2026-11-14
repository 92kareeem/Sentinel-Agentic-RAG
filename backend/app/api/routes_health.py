"""GET /healthz — no auth, no quota: load balancers and smoke tests hit this."""

from pathlib import Path

from fastapi import APIRouter

from app import __version__
from app.config import get_settings
from app.models.schemas import HealthResponse

router = APIRouter()


@router.get("/healthz")
def healthz() -> HealthResponse:
    """Liveness, the corpus version this instance serves, and the deployment
    shape a client needs to adapt its UI. Never reports a secret, only
    whether one is configured."""
    from app.rag import embeddings, index_store

    settings = get_settings()
    version = index_store.read_pointer(Path(settings.index_dir)) or "absent"

    return HealthResponse(
        status="ok",
        index_version=version,
        model_warm=embeddings._model is not None or embeddings._onnx is not None,
        storage_backend=str(settings.storage_backend),
        llm_provider=settings.llm_provider,
        llm_configured=settings.llm_configured,
        demo_mode=settings.demo_mode,
        version=__version__,
    )
