"""HTTP handlers for authentication endpoints.

All token operations go through :mod:`app.auth.token_service` which
implements the current JWT-based flow. The legacy ``legacy_auth``
module is no longer referenced here.
"""

from __future__ import annotations

from typing import Any

from app.auth.token_service import TokenService
from app.exceptions import AuthError

_token_service = TokenService()


async def login_handler(request: Any) -> dict[str, Any]:
    """Authenticate a user with email and password.

    Returns an access token and a refresh token on success.
    """
    body = await request.json()
    email: str = body.get("email", "")
    password: str = body.get("password", "")

    if not email or not password:
        raise AuthError("Email and password are required", code="MISSING_CREDENTIALS")

    tokens = _token_service.authenticate(email=email, password=password)
    if tokens is None:
        raise AuthError("Invalid credentials", code="INVALID_CREDENTIALS")

    return {
        "status": "ok",
        "access_token": tokens["access_token"],
        "refresh_token": tokens["refresh_token"],
    }


async def signup_handler(request: Any) -> dict[str, Any]:
    """Register a new user account.

    Expects ``email``, ``password``, and ``display_name`` in the body.
    """
    body = await request.json()
    email: str = body["email"]
    password: str = body["password"]
    display_name: str = body.get("display_name", "")

    user = _token_service.register_user(
        email=email,
        password=password,
        display_name=display_name,
    )
    return {"status": "created", "user_id": user["id"]}


async def refresh_token_handler(request: Any) -> dict[str, Any]:
    """Exchange a refresh token for a new access token.

    The old refresh token is invalidated after use (rotation).
    """
    body = await request.json()
    refresh_token: str = body.get("refresh_token", "")

    if not refresh_token:
        raise AuthError("Refresh token is required", code="MISSING_TOKEN")

    new_tokens = _token_service.rotate_refresh_token(refresh_token)
    return {
        "status": "ok",
        "access_token": new_tokens["access_token"],
        "refresh_token": new_tokens["refresh_token"],
    }
