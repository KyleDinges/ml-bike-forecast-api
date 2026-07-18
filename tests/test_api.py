from fastapi.testclient import TestClient

from bike_demand_api.app import app
from bike_demand_api.config import MAX_REQUEST_BYTES


def _payload(count: int = 1, actuals: bool = False) -> dict:
    records = []
    for hour in range(count):
        record = {
            "date": "2012-07-01",
            "hour": hour % 24,
            "holiday": False,
            "workingday": False,
            "weather_condition": 1,
            "temperature_normalized": 0.7,
            "feels_like_temperature_normalized": 0.7,
            "humidity_normalized": 0.5,
            "windspeed_normalized": 0.2,
        }
        if actuals:
            record["actual_demand"] = 100
        records.append(record)
    return {"records": records}


def test_forecast_returns_typed_prediction_evaluation_drift_and_request_id():
    with TestClient(app) as client:
        response = client.post("/v1/forecasts", json=_payload(actuals=True))

    assert response.status_code == 200
    assert response.headers["X-Request-ID"]
    body = response.json()
    assert body["model_metadata"]["approved_model"] in {"XGBoost", "SeasonalMedian"}
    assert body["predictions"][0]["predicted_demand"] >= 0
    assert body["evaluation"]["record_count"] == 1
    assert body["drift"]["status"] == "insufficient_sample"


def test_invalid_and_mixed_actual_demand_requests_use_the_documented_error_contract():
    mixed = _payload(count=2, actuals=True)
    del mixed["records"][1]["actual_demand"]
    with TestClient(app) as client:
        malformed = client.post("/v1/forecasts", json={"records": [{"date": "not-a-date"}]})
        mixed_actuals = client.post("/v1/forecasts", json=mixed)
        missing_route = client.get("/missing")

    for response in (malformed, mixed_actuals):
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "INVALID_REQUEST"
        assert response.headers["X-Request-ID"]
    assert missing_route.status_code == 404
    assert missing_route.json()["error"]["code"] == "NOT_FOUND"


def test_model_metadata_and_request_size_limit_are_exposed():
    with TestClient(app) as client:
        metadata = client.get("/v1/model")
        too_large = client.post(
            "/v1/forecasts",
            content=b"x" * (MAX_REQUEST_BYTES + 1),
            headers={"Content-Type": "application/json"},
        )

    assert metadata.status_code == 200
    assert metadata.json()["feature_schema_version"] == 1
    assert too_large.status_code == 413
    assert too_large.json()["error"]["code"] == "REQUEST_TOO_LARGE"
    assert too_large.headers["X-Request-ID"]


def test_artifact_unavailable_is_a_typed_503_response():
    with TestClient(app) as client:
        original_service = client.app.state.forecast_service
        original_error = client.app.state.artifact_error
        try:
            client.app.state.forecast_service = None
            client.app.state.artifact_error = "Test artifact outage."
            response = client.get("/healthz")
        finally:
            client.app.state.forecast_service = original_service
            client.app.state.artifact_error = original_error

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "ARTIFACT_UNAVAILABLE"
