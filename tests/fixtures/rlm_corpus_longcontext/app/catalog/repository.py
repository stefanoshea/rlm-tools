"""Product repository with caching.

Handles persistence and retrieval of :class:`~app.catalog.models.Product`
entities, using the Redis cache layer to reduce database load.
"""

from __future__ import annotations

import logging
from typing import Any

from app.cache.backend import RedisBackend
from app.cache.registry import PRODUCT_CACHE_TTL
from app.catalog.models import Product
from app.exceptions import CatalogError

logger = logging.getLogger(__name__)


class ProductRepository:
    """Data-access layer for products with transparent caching."""

    def __init__(self, cache: RedisBackend | None = None) -> None:
        self._cache = cache or RedisBackend()

    def _cache_key(self, product_id: int | str) -> str:
        return f"product:{product_id}"

    def get_by_id(self, product_id: int) -> Product | None:
        """Fetch a product by its primary key.

        Checks the cache first; on a miss the database is queried and
        the result is stored with :data:`PRODUCT_CACHE_TTL`.
        """
        key = self._cache_key(product_id)
        cached = self._cache.get(key)
        if cached is not None:
            logger.debug("Cache hit for product %s", product_id)
            return self._deserialize(cached)

        logger.debug("Cache miss for product %s — querying database", product_id)
        row = self._query_db(product_id)
        if row is None:
            return None
        self._cache.set(key, row, ttl=PRODUCT_CACHE_TTL)
        return self._deserialize(row)

    def search(self, query: str) -> list[Product]:
        """Full-text search across product names and descriptions."""
        results: list[dict[str, Any]] = self._search_db(query)
        return [self._deserialize(r) for r in results]

    def list_all(self, limit: int = 50, offset: int = 0) -> list[Product]:
        """Return a paginated list of products."""
        rows: list[dict[str, Any]] = self._list_db(limit, offset)
        return [self._deserialize(r) for r in rows]

    # ------------------------------------------------------------------
    # Private helpers (stubs for actual DB interaction)
    # ------------------------------------------------------------------
    @staticmethod
    def _query_db(product_id: int) -> dict[str, Any] | None:
        raise CatalogError("Database layer not configured")

    @staticmethod
    def _search_db(query: str) -> list[dict[str, Any]]:
        raise CatalogError("Database layer not configured")

    @staticmethod
    def _list_db(limit: int, offset: int) -> list[dict[str, Any]]:
        raise CatalogError("Database layer not configured")

    @staticmethod
    def _deserialize(data: dict[str, Any]) -> Product:
        return Product(**data)
