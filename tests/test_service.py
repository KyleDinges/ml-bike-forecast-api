import numpy as np

from bike_demand_api.artifacts import ApprovedArtifact, load_approved_artifact
from bike_demand_api.config import DEFAULT_ARTIFACT_PATH, DEFAULT_MANIFEST_PATH, Settings
from bike_demand_api.schemas import FeatureForecastRequest
from bike_demand_api.service import ForecastService


class NegativeModel:
    def predict(self, frame):
        return np.full(len(frame), -5.0)


def test_forecast_clamps_negative_model_predictions_to_zero():
    approved = load_approved_artifact(DEFAULT_ARTIFACT_PATH, DEFAULT_MANIFEST_PATH)
    artifact = ApprovedArtifact(
        bundle={**approved.bundle, "model": NegativeModel(), "interval_radius": 10.0},
        manifest=approved.manifest,
    )
    settings = Settings(
        artifact_path=DEFAULT_ARTIFACT_PATH,
        manifest_path=DEFAULT_MANIFEST_PATH,
        drift_min_batch_size=30,
        drift_psi_watch_threshold=0.10,
        drift_psi_alert_threshold=0.25,
    )
    request = FeatureForecastRequest.model_validate(
        {
            "records": [
                {
                    "date": "2012-07-01",
                    "hour": 14,
                    "holiday": False,
                    "workingday": False,
                    "weather_condition": 1,
                    "temperature_normalized": 0.7,
                    "feels_like_temperature_normalized": 0.7,
                    "humidity_normalized": 0.5,
                    "windspeed_normalized": 0.2,
                }
            ]
        }
    )

    response = ForecastService(artifact, settings).forecast(request)

    assert response.predictions[0].predicted_demand == 0
    assert response.predictions[0].interval_lower == 0
    assert response.predictions[0].interval_upper == 10
