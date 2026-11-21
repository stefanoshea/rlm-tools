"""Route registration for the HTTP API.

Centralises all URL-to-handler mappings so that the application
bootstrap only needs a single ``register_routes`` call.
"""

from __future__ import annotations

from typing import Any

from app.api.order_handlers import (
    cancel_order_handler,
    create_order_handler,
    get_order_handler,
)
from app.api.catalog_handlers import (
    get_product_handler,
    list_products_handler,
    search_products_handler,
)
from app.api.auth_handlers import (
    login_handler,
    refresh_token_handler,
    signup_handler,
)


def register_routes(app: Any) -> None:
    """Wire URL patterns to their handler functions.

    Parameters
    ----------
    app:
        The ASGI/WSGI application instance that supports ``add_route``.
    """
    # Order endpoints
    app.add_route("POST", "/orders", create_order_handler)
    app.add_route("GET", "/orders/{order_id}", get_order_handler)
    app.add_route("POST", "/orders/{order_id}/cancel", cancel_order_handler)

    # Catalog endpoints
    app.add_route("GET", "/products", list_products_handler)
    app.add_route("GET", "/products/{product_id}", get_product_handler)
    app.add_route("GET", "/products/search", search_products_handler)

    # Auth endpoints
    app.add_route("POST", "/auth/login", login_handler)
    app.add_route("POST", "/auth/signup", signup_handler)
    app.add_route("POST", "/auth/refresh", refresh_token_handler)
