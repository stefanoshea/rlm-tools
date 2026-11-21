# DEPRECATED: migrated to Stripe Q2 2024
# See docs/payment_integration.md for migration details
"""PayPal payment gateway adapter (deprecated).

.. deprecated::
    The application no longer processes payments through PayPal.
    Use :class:`app.payments.stripe_adapter.StripeGateway` instead.
"""

from __future__ import annotations

import json
import logging
from decimal import Decimal
from typing import Any

from app.payments.gateway import PaymentGateway
from app.exceptions import PaymentError

logger = logging.getLogger(__name__)

PAYPAL_WEBHOOK_ID: str = "WH-1AB23CD456EF789GH"
_PAYPAL_API_BASE: str = "https://api.paypal.com/v1"


class PayPalGateway(PaymentGateway):
    """PayPal implementation of the payment gateway (deprecated)."""

    def __init__(self, client_id: str = "", client_secret: str = "") -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        logger.warning("PayPalGateway is deprecated — use StripeGateway")

    def charge(self, amount: Decimal, currency: str, token: str) -> dict[str, Any]:
        raise PaymentError("PayPal charges disabled", code="PAYPAL_DISABLED")

    def refund(self, charge_id: str, amount: Decimal | None = None) -> dict[str, Any]:
        raise PaymentError("PayPal refunds disabled", code="PAYPAL_DISABLED")

    def verify_webhook(self, payload: bytes, signature: str) -> dict[str, Any]:
        """Verify PayPal webhook using webhook ID (different from Stripe)."""
        if not signature.startswith(PAYPAL_WEBHOOK_ID[:6]):
            raise PaymentError("Invalid PayPal webhook signature", code="WEBHOOK_INVALID")
        return json.loads(payload)
