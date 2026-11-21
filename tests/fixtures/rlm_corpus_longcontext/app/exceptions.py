"""Exception hierarchy for the ShipStation application.

All domain exceptions inherit from :class:`ShipStationError` so that
top-level handlers can catch the full tree with a single ``except``.
"""

from __future__ import annotations


class ShipStationError(Exception):
    """Base exception for all ShipStation errors."""

    def __init__(self, message: str = "", *, code: str | None = None) -> None:
        self.code = code
        super().__init__(message)


class PaymentError(ShipStationError):
    """Raised when a payment operation fails."""

    def __init__(
        self, message: str = "Payment processing failed", *, code: str | None = "PAYMENT_FAILED"
    ) -> None:
        super().__init__(message, code=code)


class AuthError(ShipStationError):
    """Raised for authentication or authorization failures."""

    def __init__(
        self, message: str = "Authentication failed", *, code: str | None = "AUTH_ERROR"
    ) -> None:
        super().__init__(message, code=code)


class CatalogError(ShipStationError):
    """Raised when a catalog operation (product lookup, variant update) fails."""

    def __init__(
        self, message: str = "Catalog operation failed", *, code: str | None = "CATALOG_ERROR"
    ) -> None:
        super().__init__(message, code=code)


class OrderError(ShipStationError):
    """Raised when order creation, update, or fulfillment fails."""

    def __init__(
        self, message: str = "Order operation failed", *, code: str | None = "ORDER_ERROR"
    ) -> None:
        super().__init__(message, code=code)


class NotificationError(ShipStationError):
    """Raised when a notification dispatch fails."""

    def __init__(
        self, message: str = "Notification delivery failed", *, code: str | None = "NOTIFICATION_ERROR"
    ) -> None:
        super().__init__(message, code=code)
