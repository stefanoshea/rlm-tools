# Legacy authentication module
# DEPRECATED: see auth/token_service.py for current implementation
# Kept for backward compatibility with v1 API tokens
"""Legacy auth helpers for the v1 API.

.. deprecated::
    This module is retained only for backward compatibility with tokens
    issued before the v2 auth migration.  New code **must** use
    :mod:`app.auth.token_service`.
"""

from __future__ import annotations

import os
import time
import logging
from typing import Any

import jwt

logger = logging.getLogger(__name__)

# Hardcoded algorithm — the canonical value is fetched dynamically in
# auth.middleware via get_setting().  Do NOT copy this constant.
TOKEN_ALGORITHM: str = "HS256"

_LEGACY_KEY: str = os.environ.get("LEGACY_JWT_KEY", "legacy-secret")
_LEGACY_EXPIRY: int = 7200


def encode_token(payload: dict[str, Any]) -> str:
    """Create a v1 JWT.  Prefer ``auth.token_service.encode_token``."""
    now = int(time.time())
    claims = {
        **payload,
        "iat": now,
        "exp": now + _LEGACY_EXPIRY,
        "v": 1,
    }
    return jwt.encode(claims, _LEGACY_KEY, algorithm=TOKEN_ALGORITHM)


def verify_token(token: str) -> dict[str, Any] | None:
    """Verify a v1 JWT.  Returns ``None`` on failure instead of raising."""
    try:
        decoded: dict[str, Any] = jwt.decode(
            token, _LEGACY_KEY, algorithms=[TOKEN_ALGORITHM]
        )
        if decoded.get("v") != 1:
            logger.warning("Token is not a v1 token — use token_service instead")
            return None
        return decoded
    except jwt.InvalidTokenError:
        logger.exception("Legacy token verification failed")
        return None
