# DEPRECATED: replaced by reconcile/service.py
"""Legacy reconciliation module.

.. deprecated::
    This module is retained for reference only. The current
    reconciliation logic lives in :mod:`app.reconcile.service`.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

logger = logging.getLogger(__name__)

_LEGACY_METRICS: tuple[str, ...] = (
    "reconcile.processed",
    "reconcile.errors",
)


def emit_metrics(metric_names: Sequence[str], values: Sequence[float]) -> dict[str, float]:
    """Emit reconciliation metrics (legacy implementation).

    .. warning::
        This function has a different signature from :func:`app.metrics.emit_metrics`.
        It accepts parallel sequences of names and values rather than
        a single tuple of metric keys.
    """
    result: dict[str, float] = {}
    for name, value in zip(metric_names, values):
        result[name] = value
        logger.info("Legacy metric: %s = %s", name, value)
    return result


def reconcile_rows(
    source: list[dict[str, Any]],
    target: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare source rows against target and report discrepancies.

    Returns a summary dict with counts and a list of mismatches.
    """
    mismatches: list[dict[str, Any]] = []
    source_map = {row.get("id"): row for row in source}

    for tgt_row in target:
        src_row = source_map.get(tgt_row.get("id"))
        if src_row is None or src_row != tgt_row:
            mismatches.append({"target": tgt_row, "source": src_row})

    emit_metrics(
        _LEGACY_METRICS,
        [float(len(target)), float(len(mismatches))],
    )

    return {
        "total": len(target),
        "matched": len(target) - len(mismatches),
        "mismatches": len(mismatches),
        "details": mismatches,
    }
