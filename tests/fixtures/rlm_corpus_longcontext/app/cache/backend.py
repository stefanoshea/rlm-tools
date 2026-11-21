"""Cache backend abstraction layer.

Defines the :class:`CacheBackend` interface and the default
:class:`RedisBackend` implementation used throughout the application.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any

from app.settings import REDIS_URL

logger = logging.getLogger(__name__)


class CacheBackend(ABC):
    """Abstract base class for cache backends."""

    @abstractmethod
    def get(self, key: str) -> Any | None:
        """Retrieve a value by key, returning ``None`` on cache miss."""
        ...

    @abstractmethod
    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Store a value with an optional TTL in seconds."""
        ...

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Remove a key from the cache. Returns ``True`` if the key existed."""
        ...


class RedisBackend(CacheBackend):
    """Redis-backed cache using the application's shared Redis connection."""

    def __init__(self, redis_url: str = REDIS_URL) -> None:
        import redis

        self._client = redis.from_url(redis_url, decode_responses=True)
        logger.info("RedisBackend connected to %s", redis_url)

    def get(self, key: str) -> Any | None:
        raw = self._client.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        payload = json.dumps(value)
        if ttl is not None:
            self._client.setex(key, ttl, payload)
        else:
            self._client.set(key, payload)

    def delete(self, key: str) -> bool:
        return bool(self._client.delete(key))
