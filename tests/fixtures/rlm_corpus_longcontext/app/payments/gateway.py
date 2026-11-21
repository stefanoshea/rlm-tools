"""Abstract payment gateway interface.

All concrete payment adapters must implement this protocol so
that the application can swap providers without changing domain logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any


class PaymentGateway(ABC):
    """Interface for payment provider adapters."""

    @abstractmethod
    def charge(
        self,
        amount: Decimal,
        currency: str,
        token: str,
    ) -> dict[str, Any]:
        """Create a charge and return the provider's charge object."""
        ...

    @abstractmethod
    def refund(
        self,
        charge_id: str,
        amount: Decimal | None = None,
    ) -> dict[str, Any]:
        """Refund a previous charge (full or partial)."""
        ...

    @abstractmethod
    def verify_webhook(
        self,
        payload: bytes,
        signature: str,
    ) -> dict[str, Any]:
        """Verify and decode an incoming webhook payload.

        Raises
        ------
        PaymentError
            If the signature is invalid.
        """
        ...
