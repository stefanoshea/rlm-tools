SHIPMENT_RETRY_BUDGET = 5
INVOICE_RETRY_BUDGET = 3
DEFAULT_RETRY_BUDGET = 2

RETRY_BUDGETS = {
    "shipment": SHIPMENT_RETRY_BUDGET,
    "invoice": INVOICE_RETRY_BUDGET,
}


def get_retry_budget(workflow: str) -> int:
    return RETRY_BUDGETS.get(workflow, DEFAULT_RETRY_BUDGET)
