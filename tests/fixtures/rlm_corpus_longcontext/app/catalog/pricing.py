"""Pricing engine for the product catalog.

Calculates final prices including quantity tiers and discount codes.
The discount strategy is controlled via ``settings.yaml``.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

from app.settings import get_setting
from app.catalog.models import Product

logger = logging.getLogger(__name__)

DISCOUNT_STRATEGY: str = get_setting("pricing.discount_strategy", "percentage")

_QUANTITY_TIERS: list[tuple[int, Decimal]] = [
    (100, Decimal("0.15")),
    (50, Decimal("0.10")),
    (10, Decimal("0.05")),
]


def calculate_price(product: Product, quantity: int = 1) -> Decimal:
    """Calculate the total price for *quantity* units of *product*.

    Applies tier-based volume discounts when applicable.
    """
    unit_price = product.base_price
    discount = Decimal("0")
    for threshold, pct in _QUANTITY_TIERS:
        if quantity >= threshold:
            discount = pct
            break
    discounted = unit_price * (1 - discount)
    total = discounted * quantity
    logger.debug(
        "Price: product=%s qty=%d unit=%s discount=%s total=%s",
        product.id, quantity, unit_price, discount, total,
    )
    return total


def apply_discount(price: Decimal, discount_code: str) -> Decimal:
    """Apply a promotional discount code to a price.

    Uses the configured :data:`DISCOUNT_STRATEGY` to determine how the
    code value is interpreted.
    """
    code_value = _lookup_discount_code(discount_code)
    if code_value is None:
        return price
    if DISCOUNT_STRATEGY == "percentage":
        return price * (1 - code_value / Decimal("100"))
    # flat strategy
    return max(price - code_value, Decimal("0"))


def _lookup_discount_code(code: str) -> Decimal | None:
    """Stub: look up a discount code from the promotions table."""
    logger.debug("Looking up discount code=%s", code)
    return None
