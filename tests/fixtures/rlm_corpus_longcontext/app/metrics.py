SUCCESS_METRICS = (
    "reconcile.rows_updated",
    "reconcile.latency_ms",
)

FAILURE_METRICS = (
    "reconcile.error_count",
    "reconcile.retry_count",
)


def emit_metrics(metric_keys: tuple[str, ...]) -> tuple[str, ...]:
    return metric_keys
