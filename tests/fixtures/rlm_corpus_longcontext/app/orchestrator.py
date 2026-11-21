from app.retry_config import get_retry_budget


class ShipmentOrchestrator:
    def __init__(self) -> None:
        self.retry_budget = get_retry_budget("shipment")

    def dispatch(self) -> int:
        return self.retry_budget
