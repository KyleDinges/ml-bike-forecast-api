"""Approved-artifact inference and monitoring orchestration."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .artifacts import ApprovedArtifact
from .config import Settings
from .ml import MODEL_FEATURE_COLUMNS, drift_report, enrich_features
from .schemas import FeatureForecastRequest, ForecastResponse


class ForecastService:
    def __init__(self, artifact: ApprovedArtifact, settings: Settings) -> None:
        self.artifact = artifact
        self.settings = settings

    def model_metadata(self) -> dict[str, Any]:
        manifest = self.artifact.manifest
        return {
            "model_version": manifest["model_version"],
            "feature_schema_version": manifest["feature_schema_version"],
            "approved_model": manifest["approved_model"],
            "baseline_model": manifest["baseline_model"],
            "candidate_model": manifest["candidate_model"],
            "validation_metrics": manifest["validation_metrics"],
            "promotion_rule": manifest["promotion_rule"],
            "promotion_decision": manifest["promotion_decision"],
            "promotion_reason": manifest["promotion_reason"],
            "test_metrics": manifest["test_metrics"],
            "interval": manifest["interval"],
            "active_drift_configuration": {
                "minimum_batch_size": self.settings.drift_min_batch_size,
                "psi_watch_threshold": self.settings.drift_psi_watch_threshold,
                "psi_alert_threshold": self.settings.drift_psi_alert_threshold,
                "weather_baseline_conditioning": manifest["default_drift_configuration"]["weather_baseline_conditioning"],
            },
        }

    def forecast(self, request: FeatureForecastRequest) -> ForecastResponse:
        records = [record.model_dump() for record in request.records]
        actuals = [record.pop("actual_demand") for record in records]
        frame = enrich_features(pd.DataFrame(records))
        model = self.artifact.bundle["model"]
        if self.artifact.manifest["approved_model"] == "XGBoost":
            predictions = np.asarray(model.predict(frame[MODEL_FEATURE_COLUMNS]), dtype=float)
        else:
            predictions = np.asarray(model.predict(frame), dtype=float)
        predictions = np.maximum(0, predictions)
        radius = float(self.artifact.bundle["interval_radius"])
        points = []
        for source, prediction, actual in zip(request.records, predictions, actuals, strict=True):
            points.append(
                {
                    "date": source.date,
                    "hour": source.hour,
                    "predicted_demand": float(prediction),
                    "interval_lower": float(max(0, prediction - radius)),
                    "interval_upper": float(prediction + radius),
                    "actual_demand": actual,
                    "residual": float(actual - prediction) if actual is not None else None,
                }
            )
        evaluation = None
        if actuals[0] is not None:
            actual_array = np.asarray(actuals, dtype=float)
            lower = np.maximum(0, predictions - radius)
            upper = predictions + radius
            evaluation = {
                "record_count": len(actual_array),
                "mae": float(mean_absolute_error(actual_array, predictions)),
                "rmse": float(mean_squared_error(actual_array, predictions) ** 0.5),
                "interval_coverage": float(np.mean((actual_array >= lower) & (actual_array <= upper))),
            }
        drift = drift_report(
            frame,
            self.artifact.manifest["drift_baseline"],
            self.settings.drift_min_batch_size,
            self.settings.drift_psi_watch_threshold,
            self.settings.drift_psi_alert_threshold,
        )
        return ForecastResponse(
            model_metadata=self.model_metadata(), predictions=points, evaluation=evaluation, drift=drift
        )

    @staticmethod
    def example_request() -> dict[str, Any]:
        return {
            "feature_input": {
                "records": [
                    {
                        "date": "2012-07-01",
                        "hour": 14,
                        "holiday": False,
                        "workingday": False,
                        "weather_condition": 1,
                        "temperature_normalized": 0.827,
                        "feels_like_temperature_normalized": 0.806,
                        "humidity_normalized": 0.50,
                        "windspeed_normalized": 0.2836,
                        "actual_demand": 240,
                    }
                ]
            }
        }
