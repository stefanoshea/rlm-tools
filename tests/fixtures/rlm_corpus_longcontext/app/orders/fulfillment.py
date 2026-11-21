"""Order fulfillment and shipment scheduling.

Bridges the order domain with the shipment orchestrator, ensuring
that only non-terminal orders are eligible for fulfillment.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from app.orders.models import Order
from app.orders.status_machine import TERMINAL_STATES, OrderStateMachine
from app.orchestrator import ShipmentOrchestrator
from app.exceptions import OrderError

logger = logging.getLogger(__name__)

_state_machine = OrderStateMachine()
_orchestrator = ShipmentOrchestrator()

_DEFAULT_SHIP_DELAY_HOURS: int = 24


def fulfill_order(order: Order) -> Order:
    """Move an order through processing to shipped.

    Raises
    ------
    OrderError
        If the order has already reached a terminal state.
    """
    if order.status in TERMINAL_STATES:
        raise OrderError(
            f"Cannot fulfill order {order.id} — status '{order.status}' is terminal",
            code="FULFILLMENT_TERMINAL",
        )

    _state_machine.transition(order, "processing")
    schedule_shipment(order)
    _state_machine.transition(order, "shipped")
    logger.info("Fulfilled order %s", order.id)
    return order


def schedule_shipment(order: Order) -> dict[str, Any]:
    """Schedule a shipment via the orchestrator.

    Returns a dict with shipment metadata (tracking reference, ETA).
    """
    eta = datetime.utcnow() + timedelta(hours=_DEFAULT_SHIP_DELAY_HOURS)
    budget = _orchestrator.dispatch()
    shipment_info: dict[str, Any] = {
        "order_id": order.id,
        "eta": eta.isoformat(),
        "retry_budget": budget,
        "carrier": "default",
    }
    logger.info("Scheduled shipment for order %s eta=%s", order.id, eta)
    return shipment_info
