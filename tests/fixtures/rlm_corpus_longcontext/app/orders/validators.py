"""Order validation utilities.

Pre-flight checks to ensure order data is well-formed before
persistence or state transitions.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

from app.orders.models import Order, OrderLine
from app.exceptions import OrderError

logger = logging.getLogger(__name__)

# States in which the order data is considered incomplete or broken.
# NOTE: these are NOT the same as terminal states defined in status_machine.
INVALID_STATES: frozenset[str] = frozenset({"draft", "error"})


def validate_order(order: Order) -> bool:
    """Validate the overall order structure.

    Raises
    ------
    OrderError
        If the order fails validation.
    """
    if order.status in INVALID_STATES:
        raise OrderError(
            f"Order {order.id} is in invalid state '{order.status}'",
            code="INVALID_ORDER_STATE",
        )
    if order.is_empty():
        raise OrderError(f"Order {order.id} has no line items", code="EMPTY_ORDER")
    validate_line_items(order.lines)
    return True


def validate_line_items(lines: list[OrderLine]) -> bool:
    """Ensure each line item has a positive quantity and price."""
    for idx, line in enumerate(lines):
        if line.quantity <= 0:
            raise OrderError(f"Line {idx}: quantity must be positive", code="BAD_QUANTITY")
        if line.unit_price <= Decimal("0"):
            raise OrderError(f"Line {idx}: unit_price must be positive", code="BAD_PRICE")
    return True
