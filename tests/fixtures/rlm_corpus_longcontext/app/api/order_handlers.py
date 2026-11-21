"""HTTP handlers for order operations.

Each handler receives a ``Request`` object and returns a ``Response``.
Heavy lifting is delegated to :mod:`app.orders.service`.
"""

from __future__ import annotations

from typing import Any

from app.auth.permissions import require_admin
from app.exceptions import OrderError
from app.orders.service import OrderService

_order_service = OrderService()


async def create_order_handler(request: Any) -> dict[str, Any]:
    """Create a new order from the request payload.

    Expects a JSON body with ``items``, ``customer_id``, and optional
    ``shipping_address``.
    """
    body = await request.json()
    items: list[dict[str, Any]] = body.get("items", [])
    customer_id: str = body["customer_id"]
    shipping_address: dict[str, str] | None = body.get("shipping_address")

    if not items:
        raise OrderError("Order must contain at least one item", code="EMPTY_ORDER")

    order = _order_service.create_order(
        customer_id=customer_id,
        items=items,
        shipping_address=shipping_address,
    )
    return {"status": "created", "order_id": order["id"]}


async def get_order_handler(request: Any) -> dict[str, Any]:
    """Retrieve a single order by its identifier.

    Returns 404 if the order does not exist.
    """
    order_id: str = request.path_params["order_id"]
    order = _order_service.get_order(order_id)
    if order is None:
        return {"error": "Order not found", "code": "NOT_FOUND"}
    return {"status": "ok", "order": order}


@require_admin
async def cancel_order_handler(request: Any) -> dict[str, Any]:
    """Cancel an existing order.  Admin-only.

    Applies refund logic and emits a ``order.cancelled`` event.
    """
    order_id: str = request.path_params["order_id"]
    try:
        _order_service.cancel_order(order_id)
    except OrderError as exc:
        return {"error": str(exc), "code": exc.code}
    return {"status": "cancelled", "order_id": order_id}
