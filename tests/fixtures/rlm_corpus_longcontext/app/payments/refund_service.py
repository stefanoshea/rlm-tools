"""Refund processing service.

Coordinates refunds across the payment gateway, order state machine,
notification dispatch, and metrics emission.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

from app.orders.status_machine import OrderStateMachine
from app.payments.stripe_adapter import StripeGateway
from app.notifications.dispatcher import dispatch
from app.metrics import emit_metrics
from app.exceptions import PaymentError

logger = logging.getLogger(__name__)

_state_machine = OrderStateMachine()
_gateway = StripeGateway()

_REFUND_METRICS: tuple[str, ...] = (
    "refund.amount",
    "refund.count",
)


class RefundService:
    """Handles end-to-end refund processing."""

    def process_refund(
        self,
        order_id: int,
        amount: Decimal,
        charge_id: str | None = None,
    ) -> dict[str, Any]:
        """Process a refund for an order.

        Steps:
        1. Look up the order and its associated charge.
        2. Issue the refund through the payment gateway.
        3. Transition the order to ``"refunded"`` state.
        4. Emit refund metrics.
        5. Dispatch a ``refund_issued`` notification.

        Parameters
        ----------
        order_id:
            The order to refund.
        amount:
            Refund amount.
        charge_id:
            Optional Stripe charge/payment-intent ID. If not provided
            the service will look it up from the order record.
        """
        from app.orders.service import OrderService

        order_svc = OrderService()
        order = order_svc.get_order(order_id)

        effective_charge = charge_id or self._lookup_charge(order_id)
        refund_result = _gateway.refund(effective_charge, amount)

        _state_machine.transition(order, "refunded")

        emit_metrics(_REFUND_METRICS)
        logger.info(
            "Refund processed: order=%s amount=%s refund_id=%s",
            order_id, amount, refund_result.get("id"),
        )

        dispatch("refund_issued", {
            "order_id": order_id,
            "amount": str(amount),
            "refund_id": refund_result.get("id"),
        })

        return refund_result

    @staticmethod
    def _lookup_charge(order_id: int) -> str:
        """Stub: retrieve the Stripe charge ID for an order."""
        raise PaymentError(
            f"No charge found for order {order_id}", code="CHARGE_NOT_FOUND"
        )
