"""Ledger reconciliation.

Ensures that financial ledger entries are only finalized for orders
that have reached a terminal state.
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from typing import Any

from app.orders.status_machine import TERMINAL_STATES
from app.orders.models import Order
from app.exceptions import OrderError

logger = logging.getLogger(__name__)


def finalize_ledger_entry(order: Order) -> dict[str, Any]:
    """Create a finalized ledger entry for a completed order.

    Only orders in a :data:`~app.orders.status_machine.TERMINAL_STATES`
    status may be finalized.

    Raises
    ------
    OrderError
        If the order has not yet reached a terminal state.
    """
    if order.status not in TERMINAL_STATES:
        raise OrderError(
            f"Cannot finalize ledger for order {order.id} — "
            f"status '{order.status}' is not terminal",
            code="NOT_TERMINAL",
        )

    entry = {
        "order_id": order.id,
        "amount": str(order.total),
        "status": order.status,
        "finalized_at": datetime.utcnow().isoformat(),
    }
    logger.info("Finalized ledger entry for order %s", order.id)
    return entry


class LedgerReconciler:
    """Batch reconciler for ledger entries against order records."""

    def reconcile_batch(self, orders: list[Order]) -> list[dict[str, Any]]:
        """Reconcile a batch of orders, finalizing eligible ones."""
        results: list[dict[str, Any]] = []
        for order in orders:
            if order.status in TERMINAL_STATES:
                results.append(finalize_ledger_entry(order))
            else:
                logger.debug("Skipping non-terminal order %s", order.id)
        return results
