"""FastAPI app, artifact lifecycle, safe logging, and typed error handling."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .artifacts import load_approved_artifact
from .config import MAX_REQUEST_BYTES, Settings
from .errors import ApiError, ArtifactUnavailableError
from .logging_config import configure_logging
from .routes import router
from .schemas import ErrorResponse
from .service import ForecastService


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configured_log_level = configure_logging()
    logger.info("Application logging configured log_level=%s", configured_log_level)
    settings = Settings.from_environment()
    try:
        artifact = load_approved_artifact(settings.artifact_path, settings.manifest_path)
        app.state.forecast_service = ForecastService(artifact, settings)
        app.state.artifact_error = None
    except ArtifactUnavailableError as exc:
        app.state.forecast_service = None
        app.state.artifact_error = exc.message
        logger.error("Approved artifact unavailable at startup: %s", exc.message)
    yield


app = FastAPI(
    title="Bike Demand Forecast API",
    version="1.0.0",
    description="Historical Capital Bikeshare model deployment demo with explicit UCI-compatible feature inputs.",
    lifespan=lifespan,
)
app.include_router(router)


@app.middleware("http")
async def request_logging(request: Request, call_next):
    request_id = uuid.uuid4().hex
    request.state.request_id = request_id
    started = time.perf_counter()
    request.state.error_code = "none"
    content_length = request.headers.get("content-length")
    try:
        request_is_too_large = bool(content_length) and int(content_length) > MAX_REQUEST_BYTES
    except ValueError:
        request_is_too_large = False

    if request_is_too_large:
        request.state.error_code = "REQUEST_TOO_LARGE"
        body = ErrorResponse(error={"code": "REQUEST_TOO_LARGE", "message": "The request exceeds the 1 MB limit.", "details": None})
        response = JSONResponse(body.model_dump(mode="json"), status_code=413)
    else:
        response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    level = logging.DEBUG if request.url.path == "/healthz" else logging.INFO
    logger.log(
        level,
        "HTTP request completed request_id=%s method=%s path=%s status_code=%s duration_ms=%s error_code=%s",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        round((time.perf_counter() - started) * 1000),
        request.state.error_code,
    )
    return response


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.exception_handler(ApiError)
async def handle_api_error(request: Request, error: ApiError) -> JSONResponse:
    request.state.error_code = error.code
    body = ErrorResponse(error={"code": error.code, "message": error.message, "details": error.details})
    return JSONResponse(body.model_dump(mode="json"), status_code=error.status_code)


@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, error: RequestValidationError) -> JSONResponse:
    request.state.error_code = "INVALID_REQUEST"
    body = ErrorResponse(
        error={
            "code": "INVALID_REQUEST",
            "message": "The request is invalid.",
            "details": jsonable_encoder(error.errors()),
        }
    )
    return JSONResponse(body.model_dump(mode="json"), status_code=422)


@app.exception_handler(StarletteHTTPException)
async def handle_http_error(request: Request, error: StarletteHTTPException) -> JSONResponse:
    code = "NOT_FOUND" if error.status_code == 404 else "HTTP_ERROR"
    request.state.error_code = code
    body = ErrorResponse(error={"code": code, "message": str(error.detail), "details": None})
    return JSONResponse(body.model_dump(mode="json"), status_code=error.status_code)


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, error: Exception) -> JSONResponse:
    request.state.error_code = "INTERNAL_SERVER_ERROR"
    logger.error(
        "HTTP request failed request_id=%s method=%s path=%s error_type=%s",
        getattr(request.state, "request_id", "unknown"),
        request.method,
        request.url.path,
        type(error).__name__,
    )
    body = ErrorResponse(error={"code": "INTERNAL_SERVER_ERROR", "message": "The request could not be completed.", "details": None})
    return JSONResponse(body.model_dump(mode="json"), status_code=500)
