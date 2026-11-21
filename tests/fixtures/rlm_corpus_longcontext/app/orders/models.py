"""Order domain models.

Core data structures for the order aggregate including line items.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal


@dataclass
class OrderLine:
    """A single line item within an order."""

    product_id: int
    quantity: int
    unit_price: Decimal

    @property
    def subtotal(self) -> Decimal:
        return self.unit_price * self.quantity


@dataclass
class Order:
    """Order aggregate root."""

    id: int
    customer_id: int
    status: str
    lines: list[OrderLine] = field(default_factory=list)
    total: Decimal = Decimal("0")
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime | None = None
    notes: str = ""

    @property
    def computed_total(self) -> Decimal:
        """Re-derive the order total from line items."""
        return sum((line.subtotal for line in self.lines), Decimal("0"))

    @property
    def line_count(self) -> int:
        return len(self.lines)

    def is_empty(self) -> bool:
        return self.line_count == 0

    def __post_init__(self) -> None:
        if not self.total and self.lines:
            self.total = self.computed_total
