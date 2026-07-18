import logging

from bike_demand_api.logging_config import configure_logging


def test_configure_logging_defaults_to_info_when_unset_or_blank(monkeypatch):
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    assert configure_logging() == "INFO"
    assert logging.getLogger().level == logging.INFO

    monkeypatch.setenv("LOG_LEVEL", "  ")
    assert configure_logging() == "INFO"


def test_configure_logging_uses_valid_environment_value(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "debug")

    assert configure_logging() == "DEBUG"
    assert logging.getLogger().level == logging.DEBUG


def test_configure_logging_rejects_invalid_environment_value(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "verbose")

    assert configure_logging() == "INFO"
    assert logging.getLogger().level == logging.INFO
