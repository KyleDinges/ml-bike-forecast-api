"""Pydantic request and response contracts."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FeatureInputRecord(BaseModel):
    """One complete, UCI-compatible hourly feature record."""

    model_config = ConfigDict(extra="forbid")

    date: date
    hour: int = Field(ge=0, le=23)
    holiday: bool
    workingday: bool
    weather_condition: int = Field(ge=1, le=4)
    temperature_normalized: float = Field(ge=0, le=1)
    feels_like_temperature_normalized: float = Field(ge=0, le=1)
    humidity_normalized: float = Field(ge=0, le=1)
    windspeed_normalized: float = Field(ge=0, le=1)
    actual_demand: int | None = Field(default=None, ge=0)


class FeatureForecastRequest(BaseModel):
    records: list[FeatureInputRecord] = Field(min_length=1, max_length=1_000)

    @model_validator(mode="after")
    def require_all_or_none_actual_demand(self) -> "FeatureForecastRequest":
        labeled = [record.actual_demand is not None for record in self.records]
        if any(labeled) and not all(labeled):
            raise ValueError("actual_demand must be supplied for every record or omitted for every record.")
        return self


class ForecastPoint(BaseModel):
    date: date
    hour: int
    predicted_demand: float = Field(ge=0)
    interval_lower: float = Field(ge=0)
    interval_upper: float = Field(ge=0)
    actual_demand: int | None = None
    residual: float | None = None


class DriftFeature(BaseModel):
    feature: str
    kind: Literal["numeric", "categorical"]
    psi: float | None = None
    status: Literal["stable", "watch", "drifted", "insufficient_sample"]


class DriftReport(BaseModel):
    status: Literal["stable", "watch", "drifted", "insufficient_sample"]
    sample_size: int
    minimum_sample_size: int
    features: list[DriftFeature]


class EvaluationMetrics(BaseModel):
    record_count: int
    mae: float
    rmse: float
    interval_coverage: float


class ForecastResponse(BaseModel):
    model_metadata: dict[str, Any]
    predictions: list[ForecastPoint]
    evaluation: EvaluationMetrics | None = None
    drift: DriftReport


class ErrorBody(BaseModel):
    code: str
    message: str
    details: list[dict[str, Any]] | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody
