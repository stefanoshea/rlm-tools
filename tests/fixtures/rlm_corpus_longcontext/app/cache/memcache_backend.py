# TODO: remove after Q3 migration to Redis
# DEPRECATED: use cache.backend.RedisBackend instead
"""Memcached backend implementation.

This module is **deprecated** and should not be used for new features.
The application has migrated to Redis — see :mod:`app.cache.backend`.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.cache.backend import CacheBackend

logger = logging.getLogger(__name__)

# NOTE: this value is intentionally higher than the Redis default.
# It was tuned for memcached latency characteristics and is no longer
# the authoritative TTL.  The canonical value lives in cache.registry.
PRODUCT_CACHE_TTL: int = 600


class MemcacheBackend(CacheBackend):
    """Memcached cache backend (deprecated)."""

    def __init__(self, servers: list[str] | None = None) -> None:
        import pylibmc

        self._client = pylibmc.Client(servers or ["127.0.0.1:11211"])
        logger.warning("MemcacheBackend is deprecated — migrate to RedisBackend")

    def get(self, key: str) -> Any | None:
        raw = self._client.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        payload = json.dumps(value)
        self._client.set(key, payload, time=ttl or PRODUCT_CACHE_TTL)

    def delete(self, key: str) -> bool:
        return bool(self._client.delete(key))
