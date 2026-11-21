EXPORT_PIPELINE_V2 = "export_pipeline_v2"
EXPORT_DRY_RUN = "export_dry_run"
RECONCILE_VERBOSE_LOGGING = "reconcile_verbose_logging"

_FLAGS = {
    EXPORT_PIPELINE_V2: True,
    EXPORT_DRY_RUN: False,
    RECONCILE_VERBOSE_LOGGING: False,
}


def is_enabled(flag_name: str) -> bool:
    return _FLAGS.get(flag_name, False)
