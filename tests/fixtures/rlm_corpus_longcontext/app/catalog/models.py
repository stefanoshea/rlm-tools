"""Catalog domain models.

Defines the core data structures for products, categories, and
product variants used across the catalog subsystem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any


@dataclass
class Category:
    """Product category with optional parent for hierarchical trees."""

    id: int
    name: str
    slug: str
    parent_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Variant:
    """A purchasable SKU variant of a product (size, colour, etc.)."""

    id: int
    product_id: int
    sku: str
    name: str
    price: Decimal
    stock_quantity: int = 0
    attributes: dict[str, str] = field(default_factory=dict)


@dataclass
class Product:
    """Top-level product aggregate."""

    id: int
    name: str
    description: str
    category_id: int
    base_price: Decimal
    is_active: bool = True
    variants: list[Variant] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime | None = None

    @property
    def cheapest_variant(self) -> Variant | None:
        """Return the lowest-priced variant, or ``None`` if none exist."""
        if not self.variants:
            return None
        return min(self.variants, key=lambda v: v.price)
