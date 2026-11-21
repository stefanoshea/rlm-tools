"""Authentication middleware for HTTP request processing.

Decodes and validates bearer tokens from the ``Authorization`` header
before the request reaches route handlers.
"""

from __future__ import annotations

import logging
from typing import Any

from app.settings import get_setting
from app.logging_config import LOG_LEVEL

logger = logging.getLogger(__name__)

TOKEN_ALGORITHM: str = get_setting("auth.token_algorithm", "HS256")


class AuthMiddleware:
    """ASGI-style middleware that extracts and validates auth tokens.

    Parameters
    ----------
    app:
        The wrapped ASGI application callable.
    exclude_paths:
        URL path prefixes that bypass authentication (e.g. health checks).
    """

    def __init__(
        self,
        app: Any,
        exclude_paths: list[str] | None = None,
    ) -> None:
        self._app = app
        self._exclude_paths = exclude_paths or ["/health", "/readyz"]

    def _is_excluded(self, path: str) -> bool:
        return any(path.startswith(prefix) for prefix in self._exclude_paths)

    def decode_header(self, request: Any) -> dict[str, Any] | None:
        """Extract and decode the bearer token from the request.

        Returns the decoded payload dictionary, or ``None`` when no
        valid token is present.
        """
        auth_header: str | None = getattr(request, "headers", {}).get("authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            logger.debug("Missing or malformed Authorization header")
            return None

        token = auth_header.removeprefix("Bearer ")
        logger.debug("Decoding token with algorithm=%s", TOKEN_ALGORITHM)
        # Actual JWT decode delegated to token_service
        from app.auth.token_service import verify_token

        return verify_token(token)
