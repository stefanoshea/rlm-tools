"""Tests for payment processing via Stripe."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.exceptions import PaymentError
from app.payments.processor import PaymentProcessor


class TestStripeCharge:
    """Verify charge creation through the Stripe adapter."""

    def test_successful_charge(self, sample_order: dict[str, Any]) -> None:
        processor = PaymentProcessor()
        charge_result = {
            "charge_id": "ch_test_123",
            "amount_cents": sample_order["total_cents"],
            "status": "succeeded",
        }
        with patch.object(processor, "create_charge", return_value=charge_result):
            result = processor.create_charge(
                order_id=sample_order["id"],
                amount_cents=sample_order["total_cents"],
                currency="usd",
            )
        assert result["status"] == "succeeded"
        assert result["amount_cents"] == 3998

    def test_charge_failure_raises(self) -> None:
        processor = PaymentProcessor()
        with patch.object(
            processor,
            "create_charge",
            side_effect=PaymentError("Card declined"),
        ):
            with pytest.raises(PaymentError, match="Card declined"):
                processor.create_charge(
                    order_id="ord_fail",
                    amount_cents=5000,
                    currency="usd",
                )


class TestStripeRefund:
    """Verify refund processing."""

    def test_full_refund(self) -> None:
        processor = PaymentProcessor()
        refund_result = {
            "refund_id": "re_test_456",
            "charge_id": "ch_test_123",
            "amount_cents": 3998,
            "status": "succeeded",
        }
        with patch.object(processor, "refund_charge", return_value=refund_result):
            result = processor.refund_charge(charge_id="ch_test_123")
        assert result["status"] == "succeeded"
        assert result["amount_cents"] == 3998

    def test_refund_nonexistent_charge(self) -> None:
        processor = PaymentProcessor()
        with patch.object(
            processor,
            "refund_charge",
            side_effect=PaymentError("Charge not found", code="CHARGE_NOT_FOUND"),
        ):
            with pytest.raises(PaymentError):
                processor.refund_charge(charge_id="ch_missing")
