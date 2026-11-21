# DEPRECATED: This is the pre-v2 pipeline
# The current implementation lives in pipeline.py
# This file is kept for rollback purposes only
"""Legacy export pipeline.

.. deprecated::
    Use :mod:`app.export.pipeline` instead.  This module is retained
    solely for emergency rollback to the pre-v2 export system.
"""

from __future__ import annotations

from app.export.steps import LegacyExportBuilder


def build_export_pipeline() -> LegacyExportBuilder:
    """Build the legacy (pre-v2) export pipeline.

    .. warning::
        This function **always** returns a :class:`LegacyExportBuilder`.
        It does not consult feature flags; the old pipeline had no
        conditional routing.

    Returns
    -------
    LegacyExportBuilder
        The legacy export builder instance, ready to call ``.run()``.
    """
    return LegacyExportBuilder()


def run_legacy_export() -> str:
    """Convenience wrapper that builds and immediately runs the pipeline.

    Returns
    -------
    str
        The string result from :meth:`LegacyExportBuilder.run`.
    """
    builder = build_export_pipeline()
    return builder.run()


# The v2 pipeline (pipeline.py) uses DeltaExportBuilder when the
# EXPORT_PIPELINE_V2 feature flag is enabled.  This file intentionally
# ignores that flag.
_PIPELINE_VERSION = "1.0"
