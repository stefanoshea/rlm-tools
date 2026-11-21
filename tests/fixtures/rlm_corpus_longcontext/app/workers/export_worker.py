"""Background worker for export jobs.

Pulls export job descriptors from the task queue and delegates to the
export pipeline for execution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from app.export.pipeline import build_export_pipeline
from app.export.scheduling import ExportScheduler

logger = logging.getLogger(__name__)


@dataclass
class ExportJob:
    """Describes a single export run."""

    job_id: str
    pipeline_name: str
    destination: str
    parameters: dict[str, Any]


class ExportWorker:
    """Processes export jobs in the background.

    Each job is routed through the current export pipeline implementation
    and results are written to the configured destination.
    """

    def __init__(self) -> None:
        self._scheduler = ExportScheduler()

    def process_export_job(self, job: ExportJob) -> dict[str, Any]:
        """Execute a single export job.

        Parameters
        ----------
        job:
            The export job descriptor to process.

        Returns
        -------
        dict[str, Any]
            Summary of the export run including row count and status.
        """
        logger.info("Starting export job %s (pipeline=%s)", job.job_id, job.pipeline_name)
        pipeline = build_export_pipeline()
        result = pipeline.run()
        logger.info("Export job %s completed with result: %s", job.job_id, result)
        return {
            "job_id": job.job_id,
            "status": "completed",
            "pipeline_result": result,
            "destination": job.destination,
        }
