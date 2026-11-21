"""Admin dashboard views for internal operations monitoring.

Provides aggregated statistics, recent order feeds, and reconciliation
status checks for the operations team.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.metrics import SUCCESS_METRICS, FAILURE_METRICS, emit_metrics
from app.orders.service import OrderService
from app.reconcile.service import reconcile_accounts


@dataclass
class DashboardSummary:
    """Snapshot of key business metrics."""

    total_orders: int
    revenue_cents: int
    active_users: int
    error_rate: float
    generated_at: datetime


class DashboardView:
    """Renders the admin dashboard with live data.

    Parameters
    ----------
    order_service:
        Service used to query recent order activity.
    """

    def __init__(self, order_service: OrderService | None = None) -> None:
        self._order_service = order_service or OrderService()

    def get_summary_stats(self) -> DashboardSummary:
        """Return a snapshot of current business metrics."""
        success = emit_metrics(SUCCESS_METRICS)
        failures = emit_metrics(FAILURE_METRICS)
        error_rate = len(failures) / max(len(success) + len(failures), 1)
        return DashboardSummary(
            total_orders=self._order_service.count_orders(),
            revenue_cents=self._order_service.total_revenue_cents(),
            active_users=0,
            error_rate=error_rate,
            generated_at=datetime.utcnow(),
        )

    def get_recent_orders(self, limit: int = 25) -> list[dict[str, Any]]:
        """Fetch the most recently created orders.

        Parameters
        ----------
        limit:
            Maximum number of orders to return. Defaults to 25.
        """
        return self._order_service.list_orders(limit=limit)

    def get_reconcile_status(self) -> dict[str, Any]:
        """Check whether the latest reconciliation run succeeded."""
        result = reconcile_accounts()
        return {
            "healthy": len(result) > 0,
            "metrics_emitted": result,
            "checked_at": datetime.utcnow().isoformat(),
        }
