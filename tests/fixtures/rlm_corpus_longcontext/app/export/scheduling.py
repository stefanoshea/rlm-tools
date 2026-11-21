"""Cron-based scheduling for recurring export jobs.

Wraps a simple in-memory schedule store.  Production deployments should
back this with Redis or a database table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any


@dataclass
class ScheduleEntry:
    """Represents a scheduled export pipeline run."""

    schedule_id: str
    pipeline_name: str
    cron_expr: str
    last_run: datetime | None = None
    next_run: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ExportScheduler:
    """Manages cron-like schedules for export pipelines."""

    def __init__(self) -> None:
        self._schedules: dict[str, ScheduleEntry] = {}
        self._next_id: int = 1

    def schedule_export(self, pipeline: str, cron_expr: str) -> ScheduleEntry:
        """Register a new recurring export.

        Parameters
        ----------
        pipeline:
            Name of the export pipeline to run.
        cron_expr:
            Cron expression defining the run cadence.
        """
        sid = f"sched_{self._next_id}"
        self._next_id += 1
        entry = ScheduleEntry(
            schedule_id=sid,
            pipeline_name=pipeline,
            cron_expr=cron_expr,
            next_run=datetime.utcnow() + timedelta(hours=1),
        )
        self._schedules[sid] = entry
        return entry

    def get_next_run(self, schedule_id: str) -> datetime | None:
        """Return the next scheduled run time, or ``None``."""
        entry = self._schedules.get(schedule_id)
        return entry.next_run if entry else None
