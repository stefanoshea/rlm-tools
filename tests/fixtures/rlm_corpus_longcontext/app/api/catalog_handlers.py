"""HTTP handlers for the product catalog.

Delegates to :mod:`app.catalog.repository` for data access and
:mod:`app.catalog.service` for search operations.
"""

from __future__ import annotations

from typing import Any

from app.catalog.repository import ProductRepository
from app.catalog.service import SearchService
from app.exceptions import CatalogError

_product_repo = ProductRepository()
_search_service = SearchService()


async def list_products_handler(request: Any) -> dict[str, Any]:
    """Return a paginated list of products.

    Query parameters: ``page`` (default 1), ``per_page`` (default 20).
    """
    page: int = int(request.query_params.get("page", 1))
    per_page: int = int(request.query_params.get("per_page", 20))
    products = _product_repo.list_products(page=page, per_page=per_page)
    return {"status": "ok", "page": page, "products": products}


async def get_product_handler(request: Any) -> dict[str, Any]:
    """Retrieve a single product by ID.

    Returns a 404-style payload when the product does not exist.
    """
    product_id: str = request.path_params["product_id"]
    try:
        product = _product_repo.get_product(product_id)
    except CatalogError:
        return {"error": "Product not found", "code": "NOT_FOUND"}
    return {"status": "ok", "product": product}


async def search_products_handler(request: Any) -> dict[str, Any]:
    """Full-text search across the product catalog.

    Query parameter ``q`` is required; ``limit`` caps results
    (default 50).
    """
    query: str = request.query_params.get("q", "")
    limit: int = int(request.query_params.get("limit", 50))
    if not query:
        return {"error": "Missing search query", "code": "BAD_REQUEST"}
    results = _search_service.search(query=query, limit=limit)
    return {"status": "ok", "results": results}
