"""Rebuild the approved artifact from committed public bike-sharing data."""

from __future__ import annotations

from bike_demand_api.config import DEFAULT_ARTIFACT_PATH, DEFAULT_MANIFEST_PATH, PROJECT_ROOT
from bike_demand_api.ml import train_and_save


if __name__ == "__main__":
    manifest = train_and_save(
        raw_path=PROJECT_ROOT / "data" / "raw" / "hour.csv",
        artifact_path=DEFAULT_ARTIFACT_PATH,
        manifest_path=DEFAULT_MANIFEST_PATH,
        report_path=PROJECT_ROOT / "reports" / "approved_model_evaluation.json",
    )
    print(f"Approved model: {manifest['approved_model']}")
    print(f"Promotion decision: {manifest['promotion_decision']}")
    print(f"Held-out MAE: {manifest['test_metrics']['mae']:.3f}")
