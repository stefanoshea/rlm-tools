"""Logging configuration for ShipStation.

Sets up structured JSON logging for production and a human-readable
format for local development.
"""

from __future__ import annotations

import logging
import logging.config
from typing import Any

from app.settings import get_setting


LOG_LEVEL: str = get_setting("logging.level", "INFO")
LOG_FORMAT: str = get_setting("logging.format", "json")


LOGGING_CONFIG: dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "class": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(asctime)s %(name)s %(levelname)s %(message)s",
        },
        "simple": {
            "format": "[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": LOG_FORMAT if LOG_FORMAT in ("json", "simple") else "simple",
            "stream": "ext://sys.stdout",
        },
    },
    "root": {
        "level": LOG_LEVEL,
        "handlers": ["console"],
    },
}


def setup_logging() -> None:
    """Apply the logging configuration dict."""
    logging.config.dictConfig(LOGGING_CONFIG)
