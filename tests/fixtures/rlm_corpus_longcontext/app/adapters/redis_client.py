"""Redis client adapter for caching and pub/sub.

Provides a thin wrapper around ``redis.asyncio`` that reads connection
parameters from the application settings module.
"""

from __future__ import annotations

import logging
from typing import Any

import redis.asyncio as aioredis

from app.settings import REDIS_URL, CACHE_DEFAULT_TTL, get_setting

logger = logging.getLogger(__name__)

_SOCKET_TIMEOUT: int = get_setting("redis.socket_timeout", 5)


class RedisClient:
    """Async Redis client with convenience helpers."""

    def __init__(self, url: str = REDIS_URL) -> None:
        self._url = url
        self._pool: aioredis.Redis | None = None

    async def connect(self) -> None:
        """Open the Redis connection pool."""
        logger.info("Connecting to Redis at %s", self._url)
        self._pool = aioredis.from_url(
            self._url, socket_timeout=_SOCKET_TIMEOUT, decode_responses=True
        )

    async def disconnect(self) -> None:
        """Close the connection pool gracefully."""
        if self._pool is not None:
            await self._pool.close()
            logger.info("Redis connection closed")

    async def get(self, key: str) -> str | None:
        """Fetch a value by key, returning ``None`` on miss."""
        assert self._pool is not None, "RedisClient not connected"
        return await self._pool.get(key)

    async def set(
        self, key: str, value: Any, *, ttl: int = CACHE_DEFAULT_TTL
    ) -> None:
        """Store a value with an optional TTL (seconds)."""
        assert self._pool is not None, "RedisClient not connected"
        await self._pool.set(key, value, ex=ttl)

    async def delete(self, key: str) -> None:
        """Remove a key from the store."""
        assert self._pool is not None, "RedisClient not connected"
        await self._pool.delete(key)
