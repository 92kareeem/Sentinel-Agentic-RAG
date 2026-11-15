"""FastAPI app factory + Mangum handler.

Role in architecture: assembles routers and the RFC7807 error handler.
Mangum translates API Gateway events <-> ASGI, so the identical app runs
under uvicorn locally and Lambda in production.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from mangum import Mangum

from app import __version__
from app.api import (
    routes_demo,
    routes_feedback,
    routes_health,
    routes_ingest,
    routes_insights,
    routes_query,
    routes_traces,
)
from app.config import get_settings
from app.models.schemas import Problem


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Sentinel", version=__version__)
    origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    if settings.local_mode:
        # The Vite dev server and the compose web container run on a different
        # origin than the API; production fronts both from one CloudFront
        # distribution, so the wildcard-free local list is dev-only.
        origins += ["http://localhost:5173", "http://127.0.0.1:5173"]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=sorted(set(origins)),
            allow_methods=["*"],
            allow_headers=["*"],
        )
    app.include_router(routes_health.router)
    app.include_router(routes_query.router, prefix="/v1")
    app.include_router(routes_ingest.router, prefix="/v1")
    app.include_router(routes_traces.router, prefix="/v1")
    app.include_router(routes_insights.router, prefix="/v1")
    app.include_router(routes_feedback.router, prefix="/v1")
    if settings.demo_mode:
        # Fault injection exists only where config.py allowed DEMO_MODE, which
        # is local storage. The route is not even registered elsewhere.
        app.include_router(routes_demo.router, prefix="/v1")

    @app.exception_handler(HTTPException)
    async def http_problem(request: Request, exc: HTTPException) -> JSONResponse:
        # A structured detail ({"error_code": ..., "message": ...}) is preserved
        # as fields rather than str()'d into prose: ingestion failures carry a
        # machine-readable code precisely so the client can explain what went
        # wrong, and stringifying it made that code unreadable.
        detail = exc.detail
        if isinstance(detail, dict):
            message = str(detail.get("message", "error"))
            error_code = detail.get("error_code")
        else:
            message = str(detail)
            error_code = None

        body = Problem(
            title=message,
            status=exc.status_code,
            detail=message,
            error_code=error_code,
            trace_id=getattr(request.state, "trace_id", None),
        ).model_dump()
        return JSONResponse(
            status_code=exc.status_code,
            content=body,
            headers=exc.headers,
            media_type="application/problem+json",
        )

    @app.exception_handler(RequestValidationError)
    async def validation_problem(request: Request, exc: RequestValidationError) -> JSONResponse:
        body = Problem(
            title="invalid request", status=400, detail=str(exc.errors()[:3])
        ).model_dump()
        return JSONResponse(status_code=400, content=body, media_type="application/problem+json")

    return app


app = create_app()
handler = Mangum(app)  # Lambda entrypoint: app.main.handler
