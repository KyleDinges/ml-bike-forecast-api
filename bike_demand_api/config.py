"""Non-secret runtime settings for forecasting and monitoring."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "approved" / "bike_demand_bundle.joblib"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "approved" / "artifact_manifest.json"
MAX_REQUEST_BYTES = 1 * 1024 * 1024


@dataclass(frozen=True)
class Settings:
    artifact_path: Path
    manifest_path: Path
    drift_min_batch_size: int
    drift_psi_watch_threshold: float
    drift_psi_alert_threshold: float

    @classmethod
    def from_environment(cls) -> "Settings":
        settings = cls(
            artifact_path=Path(os.getenv("MODEL_ARTIFACT_PATH", str(DEFAULT_ARTIFACT_PATH))),
            manifest_path=Path(os.getenv("MODEL_MANIFEST_PATH", str(DEFAULT_MANIFEST_PATH))),
            drift_min_batch_size=int(os.getenv("DRIFT_MIN_BATCH_SIZE", "30")),
            drift_psi_watch_threshold=float(os.getenv("DRIFT_PSI_WATCH_THRESHOLD", "0.10")),
            drift_psi_alert_threshold=float(os.getenv("DRIFT_PSI_ALERT_THRESHOLD", "0.25")),
        )
        if settings.drift_min_batch_size < 1:
            raise ValueError("DRIFT_MIN_BATCH_SIZE must be at least 1.")
        if not 0 < settings.drift_psi_watch_threshold < settings.drift_psi_alert_threshold:
            raise ValueError("Drift PSI thresholds must satisfy 0 < watch < alert.")
        return settings
