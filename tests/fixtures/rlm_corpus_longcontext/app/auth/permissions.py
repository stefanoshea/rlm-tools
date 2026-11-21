"""Role-based permission checks.

Provides utilities for verifying that a token's role satisfies
the minimum required permission level.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable

from app.auth.token_service import verify_token
from app.exceptions import AuthError

logger = logging.getLogger(__name__)

ROLE_HIERARCHY: dict[str, int] = {
    "viewer": 0,
    "editor": 1,
    "manager": 2,
    "admin": 3,
}


def check_permission(token: str, required_role: str) -> dict[str, Any]:
    """Verify the token and ensure the subject has the required role.

    Returns the decoded claims on success.

    Raises
    ------
    AuthError
        If the token is invalid or the role is insufficient.
    """
    claims = verify_token(token)
    user_role: str = claims.get("role", "viewer")
    required_level = ROLE_HIERARCHY.get(required_role, 0)
    user_level = ROLE_HIERARCHY.get(user_role, 0)

    if user_level < required_level:
        logger.warning("Permission denied: user_role=%s required=%s", user_role, required_role)
        raise AuthError(
            f"Role '{user_role}' lacks '{required_role}' permission",
            code="INSUFFICIENT_ROLE",
        )
    return claims


def require_admin(handler: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator that restricts a handler to admin users only."""

    @functools.wraps(handler)
    def wrapper(request: Any, *args: Any, **kwargs: Any) -> Any:
        token = getattr(request, "token", "")
        check_permission(token, "admin")
        return handler(request, *args, **kwargs)

    return wrapper
