"""Stripe payment gateway adapter.

Implements :class:`~app.payments.gateway.PaymentGateway` using the
Stripe API.  Webhook verification uses HMAC-SHA256 with the
application's configured webhook secret.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from decimal import Decimal
from typing import Any

from app.payments.gateway import PaymentGateway
from app.settings import STRIPE_WEBHOOK_SECRET
from app.exceptions import PaymentError

logger = logging.getLogger(__name__)

# Hash algorithm used for Stripe webhook signature verification.
# This is NOT related to auth token algorithms — it is specific to
# Stripe's ``Stripe-Signature`` header scheme.
TOKEN_HASH_ALGO: str = "sha256"


class StripeGateway(PaymentGateway):
    """Stripe implementation of the payment gateway."""

    def __init__(self, api_key: str | None = None) -> None:
        import stripe

        if api_key:
            stripe.api_key = api_key
        self._stripe = stripe

    def charge(
        self,
        amount: Decimal,
        currency: str,
        token: str,
    ) -> dict[str, Any]:
        """Create a Stripe PaymentIntent and confirm it."""
        intent = self._stripe.PaymentIntent.create(
            amount=int(amount * 100),
            currency=currency.lower(),
            payment_method=token,
            confirm=True,
        )
        logger.info("Stripe charge created: %s", intent["id"])
        return dict(intent)

    def refund(
        self,
        charge_id: str,
        amount: Decimal | None = None,
    ) -> dict[str, Any]:
        """Issue a Stripe refund."""
        params: dict[str, Any] = {"payment_intent": charge_id}
        if amount is not None:
            params["amount"] = int(amount * 100)
        refund = self._stripe.Refund.create(**params)
        logger.info("Stripe refund issued: %s", refund["id"])
        return dict(refund)

    def verify_webhook(
        self,
        payload: bytes,
        signature: str,
    ) -> dict[str, Any]:
        """Verify a Stripe webhook signature using HMAC."""
        expected = hmac.new(
            STRIPE_WEBHOOK_SECRET.encode(),
            payload,
            hashlib.new(TOKEN_HASH_ALGO),
        ).hexdigest()

        if not hmac.compare_digest(expected, signature):
            raise PaymentError("Invalid Stripe webhook signature", code="WEBHOOK_INVALID")

        return json.loads(payload)
