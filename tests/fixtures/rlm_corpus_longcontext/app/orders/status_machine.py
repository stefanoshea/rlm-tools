"""Order state machine.

Enforces valid state transitions and prevents mutation of orders
that have reached a terminal state.
"""

from __future__ import annotations

import logging
from typing import Any

from app.orders.models import Order
from app.exceptions import OrderError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Terminal states — no further transitions are allowed once reached
# ---------------------------------------------------------------------------
TERMINAL_STATES: frozenset[str] = frozenset({"delivered", "cancelled", "refunded"})

# ---------------------------------------------------------------------------
# Allowed transitions: {current_state: {valid_next_states}}
# ---------------------------------------------------------------------------
_TRANSITIONS: dict[str, frozenset[str]] = {
    "draft": frozenset({"pending"}),
    "pending": frozenset({"confirmed", "cancelled"}),
    "confirmed": frozenset({"processing", "cancelled"}),
    "processing": frozenset({"shipped", "cancelled"}),
    "shipped": frozenset({"delivered", "returned"}),
    "returned": frozenset({"refunded"}),
    "delivered": frozenset(),
    "cancelled": frozenset(),
    "refunded": frozenset(),
}


class OrderStateMachine:
    """Validates and executes order state transitions."""

    def transition(self, order: Order, to_state: str) -> Order:
        """Attempt to move *order* to *to_state*.

        Parameters
        ----------
        order:
            The order aggregate to transition.
        to_state:
            The desired target state.

        Returns
        -------
        Order
            The order with its ``status`` updated.

        Raises
        ------
        OrderError
            If the transition is not permitted.
        """
        current = order.status

        if current in TERMINAL_STATES:
            raise OrderError(
                f"Order {order.id} is in terminal state '{current}' — no further transitions",
                code="TERMINAL_STATE",
            )

        allowed = _TRANSITIONS.get(current, frozenset())
        if to_state not in allowed:
            raise OrderError(
                f"Transition '{current}' -> '{to_state}' is not allowed for order {order.id}",
                code="INVALID_TRANSITION",
            )

        logger.info("Order %s: %s -> %s", order.id, current, to_state)
        order.status = to_state
        return order
