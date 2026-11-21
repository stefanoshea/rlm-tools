from app.export.pipeline import build_export_pipeline
from app.orchestrator import ShipmentOrchestrator
from app.reconcile.service import reconcile_accounts


def boot() -> None:
    pipeline = build_export_pipeline()
    orchestrator = ShipmentOrchestrator()
    pipeline.run()
    orchestrator.dispatch()
    reconcile_accounts()
