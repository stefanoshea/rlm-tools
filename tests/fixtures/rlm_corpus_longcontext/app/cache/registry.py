"""Cache key registry and TTL configuration.

Centralizes all cache key naming conventions and time-to-live values
so they can be referenced consistently across modules.
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# TTL constants (seconds)
# ---------------------------------------------------------------------------
PRODUCT_CACHE_TTL: Final[int] = 300
ORDER_CACHE_TTL: Final[int] = 60
SESSION_CACHE_TTL: Final[int] = 1800

# ---------------------------------------------------------------------------
# Internal registry mapping logical names to TTL values
# ---------------------------------------------------------------------------
_CACHE_KEYS: dict[str, int] = {
    "product": PRODUCT_CACHE_TTL,
    "order": ORDER_CACHE_TTL,
    "session": SESSION_CACHE_TTL,
}


def get_ttl(cache_name: str) -> int:
    """Return the TTL for a named cache bucket.

    Parameters
    ----------
    cache_name:
        Logical cache name (e.g. ``"product"``, ``"order"``).

    Returns
    -------
    int
        TTL in seconds. Falls back to 120 if the name is unknown.
    """
    from app.settings import CACHE_DEFAULT_TTL

    return _CACHE_KEYS.get(cache_name, CACHE_DEFAULT_TTL)
