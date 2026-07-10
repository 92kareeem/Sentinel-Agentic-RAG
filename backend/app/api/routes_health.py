"""GET /healthz — no auth, no quota: load balancers and smoke tests hit this."""

from pathlib import Path

from fastapi import APIRouter

from app.config import get_settings
from app.models.schemas import HealthResponse

router = APIRouter()


@router.get("/healthz")
def healthz() -> HealthResponse:
    settings = get_settings()
    index_file = Path(settings.index_dir) / "faiss.index"
    version = "absent"
    if index_file.exists():
        version = str(int(index_file.stat().st_mtime))

    from app.rag import embeddings

    return HealthResponse(
        status="ok",
        index_version=version,
        model_warm=embeddings._model is not None,
    )
