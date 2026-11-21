"""Reconciliation reporting.

Generates summary reports for reconciliation runs, referencing
the application metrics for success/failure tracking.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

from app.metrics import SUCCESS_METRICS, emit_metrics

logger = logging.getLogger(__name__)


def generate_reconcile_report(
    date_range: tuple[date, date],
) -> dict[str, Any]:
    """Generate a reconciliation summary report.

    Parameters
    ----------
    date_range:
        A ``(start_date, end_date)`` tuple defining the report window.

    Returns
    -------
    dict[str, Any]
        Report payload including row counts and metric references.
    """
    start, end = date_range
    logger.info("Generating reconcile report for %s to %s", start, end)

    # Emit success metrics for the report run
    emitted = emit_metrics(SUCCESS_METRICS)

    report: dict[str, Any] = {
        "period_start": start.isoformat(),
        "period_end": end.isoformat(),
        "generated_at": datetime.utcnow().isoformat(),
        "metrics_emitted": list(emitted),
        "rows_processed": 0,
        "errors": 0,
    }
    logger.info("Reconcile report generated: %s", report["generated_at"])
    return report
