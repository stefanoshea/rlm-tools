"""Middleware ordering and composition for the HTTP stack.

Middleware is applied in the order defined by :data:`MIDDLEWARE_ORDER`.
Each entry is instantiated and wrapped around the application.
"""

from __future__ import annotations

from typing import Any, Callable

from app.auth.middleware import AuthMiddleware

# Middleware classes applied outermost-first.  The first entry wraps
# the entire stack; the last entry sits closest to the handler.
MIDDLEWARE_ORDER: list[str] = [
    "RequestIdMiddleware",
    "LoggingMiddleware",
    "AuthMiddleware",
    "RateLimitMiddleware",
    "CorsMiddleware",
]

_MIDDLEWARE_REGISTRY: dict[str, type] = {
    "AuthMiddleware": AuthMiddleware,
}


def build_middleware_chain(app: Any) -> Any:
    """Wrap *app* with each middleware in :data:`MIDDLEWARE_ORDER`.

    Parameters
    ----------
    app:
        The inner ASGI/WSGI application to wrap.

    Returns
    -------
    Any
        The outermost middleware instance, ready to receive requests.
    """
    wrapped = app
    for name in reversed(MIDDLEWARE_ORDER):
        middleware_cls = _MIDDLEWARE_REGISTRY.get(name)
        if middleware_cls is not None:
            wrapped = middleware_cls(wrapped)
    return wrapped
