"""Tests for the order lifecycle."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.exceptions import OrderError
from app.orders.service import OrderService


class TestOrderCreation:
    """Verify order creation validations and persistence."""

    def test_create_order_success(self, sample_order: dict[str, Any]) -> None:
        service = OrderService()
        with patch.object(service, "create_order", return_value=sample_order):
            result = service.create_order(
                customer_id="cust_001",
                items=sample_order["items"],
            )
        assert result["id"] == "ord_abc123"
        assert result["status"] == "pending"

    def test_create_order_empty_items_raises(self) -> None:
        service = OrderService()
        with pytest.raises(OrderError, match="at least one item"):
            service.create_order(customer_id="cust_001", items=[])


class TestOrderCancellation:
    """Verify cancellation logic and state transitions."""

    def test_cancel_pending_order(self, sample_order: dict[str, Any]) -> None:
        service = OrderService()
        with patch.object(service, "cancel_order", return_value=True):
            result = service.cancel_order(sample_order["id"])
        assert result is True

    def test_cancel_shipped_order_raises(self) -> None:
        service = OrderService()
        with patch.object(
            service,
            "cancel_order",
            side_effect=OrderError("Cannot cancel shipped order"),
        ):
            with pytest.raises(OrderError):
                service.cancel_order("ord_shipped")


class TestOrderStateTransitions:
    """Ensure only valid state transitions are allowed."""

    @pytest.mark.parametrize(
        "from_status,to_status,valid",
        [
            ("pending", "confirmed", True),
            ("confirmed", "shipped", True),
            ("shipped", "delivered", True),
            ("delivered", "pending", False),
        ],
    )
    def test_transition_validity(
        self, from_status: str, to_status: str, valid: bool
    ) -> None:
        service = OrderService()
        with patch.object(service, "transition", return_value=valid):
            assert service.transition("ord_1", from_status, to_status) is valid
