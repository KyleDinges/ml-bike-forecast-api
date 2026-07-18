import json

import joblib
import pytest

from bike_demand_api.artifacts import load_approved_artifact
from bike_demand_api.errors import ArtifactUnavailableError


def test_artifact_loader_rejects_missing_and_inconsistent_files(tmp_path):
    with pytest.raises(ArtifactUnavailableError):
        load_approved_artifact(tmp_path / "missing.joblib", tmp_path / "missing.json")

    artifact_path = tmp_path / "artifact.joblib"
    manifest_path = tmp_path / "manifest.json"
    joblib.dump({"approved_model": "XGBoost"}, artifact_path)
    manifest_path.write_text(
        json.dumps({"approved_model": "SeasonalMedian", "feature_schema_version": 1}), encoding="utf-8"
    )

    with pytest.raises(ArtifactUnavailableError, match="disagree"):
        load_approved_artifact(artifact_path, manifest_path)
