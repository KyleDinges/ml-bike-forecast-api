"""Typed application errors for predictable API responses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ApiError(Exception):
    code: str
    message: str
    status_code: int
    details: list[dict[str, Any]] | None = None


class ArtifactUnavailableError(ApiError):
    def __init__(self, message: str = "The approved model artifact is unavailable.") -> None:
        super().__init__("ARTIFACT_UNAVAILABLE", message, 503)
