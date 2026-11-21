# Replaced by pricing.py - kept for audit trail
"""Legacy pricing engine (v1).

.. deprecated::
    This module used flat discounts.  The current implementation in
    :mod:`app.catalog.pricing` supports configurable strategies.
"""

from __future__ import annotations

import logging
from decimal import Decimal

from app.catalog.models import Product

logger = logging.getLogger(__name__)

_FLAT_DISCOUNT_AMOUNT: Decimal = Decimal("5.00")


def calculate_price(product: Product, quantity: int = 1) -> Decimal:
    """Calculate price using the legacy flat-discount approach.

    Unlike the current engine this applies a fixed dollar-amount
    discount per unit for orders over 10 items.
    """
    unit_price = product.base_price
    if quantity >= 10:
        unit_price = max(unit_price - _FLAT_DISCOUNT_AMOUNT, Decimal("0"))
    total = unit_price * quantity
    logger.debug("Legacy price: product=%s qty=%d total=%s", product.id, quantity, total)
    return total


def apply_discount(price: Decimal, discount_code: str) -> Decimal:
    """Apply a flat discount regardless of strategy setting."""
    code_value = _lookup_code(discount_code)
    if code_value is None:
        return price
    return max(price - code_value, Decimal("0"))


def _lookup_code(code: str) -> Decimal | None:
    logger.debug("Legacy discount lookup code=%s", code)
    return None
