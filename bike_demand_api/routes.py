"""FastAPI routes kept thin by delegating to the forecast service."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Request

from .errors import ArtifactUnavailableError
from .schemas import ErrorResponse, FeatureForecastRequest, ForecastResponse
from .service import ForecastService


ERROR_RESPONSE = {"model": ErrorResponse}
router = APIRouter(responses={500: ERROR_RESPONSE, 503: ERROR_RESPONSE})
logger = logging.getLogger(__name__)


def get_forecast_service(request: Request) -> ForecastService:
    """Return the lifespan-created service or expose its startup failure as a 503."""
    service = getattr(request.app.state, "forecast_service", None)
    if service is None:
        raise ArtifactUnavailableError(getattr(request.app.state, "artifact_error", "The approved model artifact is unavailable."))
    return service


ForecastServiceDep = Annotated[ForecastService, Depends(get_forecast_service)]


@router.get("/healthz")
async def healthz(_service: ForecastServiceDep) -> dict[str, str]:
    return {"status": "ok"}


@router.get("/v1/model")
async def get_model(service: ForecastServiceDep) -> dict:
    return service.model_metadata()


@router.get("/v1/example-request")
async def get_example_request(service: ForecastServiceDep) -> dict:
    return service.example_request()


@router.post(
    "/v1/forecasts",
    response_model=ForecastResponse,
    responses={413: ERROR_RESPONSE, 422: ERROR_RESPONSE},
)
async def post_forecasts(
    payload: FeatureForecastRequest, request: Request, service: ForecastServiceDep
) -> ForecastResponse:
    logger.info(
        "POST /v1/forecasts request accepted request_id=%s mode=feature_input record_count=%s actuals_supplied=%s",
        getattr(request.state, "request_id", "unknown"),
        len(payload.records),
        payload.records[0].actual_demand is not None,
    )
    response = service.forecast(payload)
    logger.info(
        "POST /v1/forecasts completed request_id=%s mode=feature_input record_count=%s evaluation_available=%s drift_status=%s",
        getattr(request.state, "request_id", "unknown"),
        len(payload.records),
        response.evaluation is not None,
        response.drift.status,
    )
    return response
