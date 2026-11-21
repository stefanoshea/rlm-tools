"""Order service layer.

Orchestrates order creation, cancellation, and retrieval, coordinating
between the state machine, persistence, and notification dispatch.
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from typing import Any

from app.orders.models import Order, OrderLine
from app.orders.status_machine import OrderStateMachine
from app.notifications.dispatcher import dispatch
from app.exceptions import OrderError

logger = logging.getLogger(__name__)

_state_machine = OrderStateMachine()
_next_id: int = 1000


class OrderService:
    """Application-level order operations."""

    def create_order(
        self,
        customer_id: int,
        lines: list[dict[str, Any]],
    ) -> Order:
        """Create a new order and dispatch a confirmation notification.

        Parameters
        ----------
        customer_id:
            The customer placing the order.
        lines:
            List of dicts with keys ``product_id``, ``quantity``, ``unit_price``.
        """
        global _next_id
        order_lines = [
            OrderLine(
                product_id=ln["product_id"],
                quantity=ln["quantity"],
                unit_price=Decimal(str(ln["unit_price"])),
            )
            for ln in lines
        ]
        order = Order(
            id=_next_id,
            customer_id=customer_id,
            status="draft",
            lines=order_lines,
        )
        _next_id += 1

        # Move through draft -> pending -> confirmed
        _state_machine.transition(order, "pending")
        _state_machine.transition(order, "confirmed")

        dispatch("order_confirmed", {
            "order_id": order.id,
            "customer_id": customer_id,
            "total": str(order.total),
        })
        logger.info("Created order %s for customer %s", order.id, customer_id)
        return order

    def cancel_order(self, order_id: int) -> Order:
        """Cancel an existing order."""
        order = self.get_order(order_id)
        _state_machine.transition(order, "cancelled")
        dispatch("order_cancelled", {"order_id": order.id})
        logger.info("Cancelled order %s", order_id)
        return order

    def get_order(self, order_id: int) -> Order:
        """Retrieve an order by ID.

        Raises
        ------
        OrderError
            If the order is not found.
        """
        # Stub — would query a database in production
        raise OrderError(f"Order {order_id} not found", code="ORDER_NOT_FOUND")
