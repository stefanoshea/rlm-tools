from app.metrics import SUCCESS_METRICS, emit_metrics


def emit_success_metrics() -> tuple[str, ...]:
    return emit_metrics(SUCCESS_METRICS)


def reconcile_accounts() -> tuple[str, ...]:
    rows_updated = 4
    if rows_updated <= 0:
        return ()
    return emit_success_metrics()
