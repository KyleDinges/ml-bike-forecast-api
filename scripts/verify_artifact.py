"""Verify that the committed artifact and manifest can be loaded together."""

from __future__ import annotations

from bike_demand_api.artifacts import load_approved_artifact
from bike_demand_api.config import DEFAULT_ARTIFACT_PATH, DEFAULT_MANIFEST_PATH


if __name__ == "__main__":
    artifact = load_approved_artifact(DEFAULT_ARTIFACT_PATH, DEFAULT_MANIFEST_PATH)
    print(f"verified approved_model={artifact.manifest['approved_model']} version={artifact.manifest['model_version']}")
