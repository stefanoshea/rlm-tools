"""Cache invalidation helpers.

Provides convenience functions that remove stale entries from the
cache when domain objects are mutated.
"""

from __future__ import annotations

import logging

from app.cache.backend import RedisBackend
from app.cache.registry import PRODUCT_CACHE_TTL, ORDER_CACHE_TTL

logger = logging.getLogger(__name__)

_backend = RedisBackend()


def _cache_key(prefix: str, identifier: str | int) -> str:
    """Build a namespaced cache key."""
    return f"{prefix}:{identifier}"


def invalidate_product_cache(product_id: int | str) -> None:
    """Remove a product entry from the cache."""
    key = _cache_key("product", product_id)
    deleted = _backend.delete(key)
    logger.info(
        "Invalidated product cache key=%s deleted=%s ttl_was=%d",
        key, deleted, PRODUCT_CACHE_TTL,
    )


def invalidate_order_cache(order_id: int | str) -> None:
    """Remove an order entry from the cache."""
    key = _cache_key("order", order_id)
    deleted = _backend.delete(key)
    logger.info(
        "Invalidated order cache key=%s deleted=%s ttl_was=%d",
        key, deleted, ORDER_CACHE_TTL,
    )
