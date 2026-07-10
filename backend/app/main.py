"""FastAPI app factory + Mangum handler.

Role in architecture: assembles routers and the RFC7807 error handler.
Mangum translates API Gateway events <-> ASGI, so the identical app runs
under uvicorn locally and Lambda in production.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from mangum import Mangum

from app.api import routes_health, routes_ingest, routes_query, routes_traces
from app.models.schemas import Problem


def create_app() -> FastAPI:
    app = FastAPI(title="Sentinel", version="0.1.0")
    app.include_router(routes_health.router)
    app.include_router(routes_query.router, prefix="/v1")
    app.include_router(routes_ingest.router, prefix="/v1")
    app.include_router(routes_traces.router, prefix="/v1")

    @app.exception_handler(HTTPException)
    async def http_problem(request: Request, exc: HTTPException) -> JSONResponse:
        body = Problem(
            title=exc.detail if isinstance(exc.detail, str) else "error",
            status=exc.status_code,
            detail=str(exc.detail),
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
