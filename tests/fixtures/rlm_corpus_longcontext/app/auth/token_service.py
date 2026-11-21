"""Token encoding, verification, and refresh logic.

All JWT operations flow through this module so that the signing key
and algorithm are managed in a single place.
"""

from __future__ import annotations

import os
import time
import logging
from typing import Any

import jwt

from app.auth.middleware import TOKEN_ALGORITHM
from app.exceptions import AuthError

logger = logging.getLogger(__name__)

_SIGNING_KEY: str = os.environ.get("JWT_SIGNING_KEY", "change-me-in-production")
_TOKEN_EXPIRY_SECONDS: int = 3600
_REFRESH_WINDOW_SECONDS: int = 300


def encode_token(payload: dict[str, Any]) -> str:
    """Create a signed JWT from the given payload.

    Automatically injects ``iat`` and ``exp`` claims.
    """
    now = int(time.time())
    claims = {
        **payload,
        "iat": now,
        "exp": now + _TOKEN_EXPIRY_SECONDS,
    }
    token: str = jwt.encode(claims, _SIGNING_KEY, algorithm=TOKEN_ALGORITHM)
    logger.info("Encoded token for sub=%s algorithm=%s", payload.get("sub"), TOKEN_ALGORITHM)
    return token


def verify_token(token: str) -> dict[str, Any]:
    """Decode and verify a JWT, returning the claims dict.

    Raises
    ------
    AuthError
        If the token is expired, tampered with, or otherwise invalid.
    """
    try:
        decoded: dict[str, Any] = jwt.decode(
            token, _SIGNING_KEY, algorithms=[TOKEN_ALGORITHM]
        )
        return decoded
    except jwt.ExpiredSignatureError:
        raise AuthError("Token has expired", code="TOKEN_EXPIRED")
    except jwt.InvalidTokenError as exc:
        raise AuthError(f"Invalid token: {exc}", code="TOKEN_INVALID")


def refresh_token(token: str) -> str:
    """Issue a fresh token if the existing one is within the refresh window.

    Raises
    ------
    AuthError
        If the token cannot be refreshed (e.g. already expired).
    """
    claims = verify_token(token)
    remaining = claims.get("exp", 0) - int(time.time())
    if remaining > _REFRESH_WINDOW_SECONDS:
        raise AuthError("Token is not yet eligible for refresh", code="REFRESH_TOO_EARLY")
    refreshed_payload = {k: v for k, v in claims.items() if k not in ("iat", "exp")}
    return encode_token(refreshed_payload)
