"""Shared test fixtures for the ShipStation test suite."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest


@pytest.fixture()
def mock_settings() -> dict[str, Any]:
    """Minimal settings dictionary for unit tests."""
    return {
        "database": {"url": "sqlite:///:memory:"},
        "redis": {"url": "redis://localhost:6379/0"},
        "stripe": {"webhook_secret": "whsec_test_secret"},
        "auth": {"token_algorithm": "HS512"},
        "cache": {"default_ttl": 60},
    }


@pytest.fixture()
def mock_database() -> MagicMock:
    """Mocked database connection."""
    db = MagicMock()
    db.execute.return_value = None
    db.fetchone.return_value = {}
    db.fetchall.return_value = []
    return db


@pytest.fixture()
def sample_order() -> dict[str, Any]:
    """A representative order payload for testing."""
    return {
        "id": "ord_abc123",
        "customer_id": "cust_001",
        "status": "pending",
        "items": [
            {"product_id": "prod_42", "quantity": 2, "unit_price_cents": 1999},
        ],
        "total_cents": 3998,
    }


@pytest.fixture()
def sample_product() -> dict[str, Any]:
    """A representative product payload for testing."""
    return {
        "id": "prod_42",
        "name": "Widget Pro",
        "price_cents": 1999,
        "category": "widgets",
        "in_stock": True,
    }
