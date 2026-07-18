"""Operational logging configuration with no request-content logging."""

from __future__ import annotations

import logging
import os


DEFAULT_LOG_LEVEL = "INFO"
VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


def configure_logging() -> str:
    """Configure the root logger from the non-secret LOG_LEVEL setting."""
    configured_name = os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL).strip().upper()
    level_name = configured_name if configured_name in VALID_LOG_LEVELS else DEFAULT_LOG_LEVEL
    root = logging.getLogger()
    root.setLevel(getattr(logging, level_name))
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        root.addHandler(handler)
    return level_name
