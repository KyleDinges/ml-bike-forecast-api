"""Approved artifact loading and compatibility checks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from .errors import ArtifactUnavailableError


@dataclass
class ApprovedArtifact:
    bundle: dict[str, Any]
    manifest: dict[str, Any]


def load_approved_artifact(artifact_path: Path, manifest_path: Path) -> ApprovedArtifact:
    if not artifact_path.exists() or not manifest_path.exists():
        raise ArtifactUnavailableError()
    bundle = joblib.load(artifact_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if bundle.get("approved_model") != manifest.get("approved_model"):
        raise ArtifactUnavailableError("The artifact and manifest disagree about the approved model.")
    if manifest.get("feature_schema_version") != 1:
        raise ArtifactUnavailableError("The artifact feature schema is unsupported.")
    return ApprovedArtifact(bundle=bundle, manifest=manifest)
