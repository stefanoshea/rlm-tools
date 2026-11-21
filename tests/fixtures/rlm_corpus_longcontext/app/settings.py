"""Application settings loaded from settings.yaml.

Provides a centralized configuration layer. All modules should read
configuration through ``get_setting`` or the pre-defined constants
rather than parsing YAML directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


_CONFIG_PATH = Path(__file__).resolve().parent.parent / "settings.yaml"
_settings_cache: dict[str, Any] | None = None


@dataclass(frozen=True)
class AppSettings:
    """Typed representation of the top-level settings.yaml structure."""

    stripe: dict[str, Any] = field(default_factory=dict)
    auth: dict[str, Any] = field(default_factory=dict)
    database: dict[str, Any] = field(default_factory=dict)
    redis: dict[str, Any] = field(default_factory=dict)
    cache: dict[str, Any] = field(default_factory=dict)
    notifications: dict[str, Any] = field(default_factory=dict)
    logging: dict[str, Any] = field(default_factory=dict)
    feature_toggles: dict[str, bool] = field(default_factory=dict)
    rate_limits: dict[str, int] = field(default_factory=dict)


def load_settings(path: Path | None = None) -> dict[str, Any]:
    """Load and cache the YAML settings file.

    Parameters
    ----------
    path:
        Override path to settings file. Defaults to the repo-root
        ``settings.yaml``.

    Returns
    -------
    dict[str, Any]
        Parsed settings dictionary.
    """
    global _settings_cache
    if _settings_cache is not None:
        return _settings_cache

    config_path = path or Path(os.getenv("SHIPSTATION_CONFIG", str(_CONFIG_PATH)))
    with open(config_path, "r") as fh:
        _settings_cache = yaml.safe_load(fh)
    return _settings_cache


def get_setting(dotted_key: str, default: Any = None) -> Any:
    """Retrieve a nested setting using dot-notation.

    Example::

        get_setting("stripe.webhook_secret")
        get_setting("notifications.channel_map.order_confirmed", "email")
    """
    data = load_settings()
    keys = dotted_key.split(".")
    current: Any = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


# ---------------------------------------------------------------------------
# Convenience constants used across the application
# ---------------------------------------------------------------------------
STRIPE_WEBHOOK_SECRET: str = get_setting("stripe.webhook_secret")
NOTIFICATION_CHANNEL_MAP: dict[str, str] = get_setting("notifications.channel_map", {})
DATABASE_URL: str = get_setting("database.url")
REDIS_URL: str = get_setting("redis.url", "redis://localhost:6379/0")
AUTH_TOKEN_ALGORITHM: str = get_setting("auth.token_algorithm", "HS512")
CACHE_DEFAULT_TTL: int = get_setting("cache.default_ttl", 120)
