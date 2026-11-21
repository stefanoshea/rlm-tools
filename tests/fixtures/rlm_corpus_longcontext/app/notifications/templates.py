"""Notification template rendering.

Maps event types to message templates and renders them with
the provided context variables.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_TEMPLATES: dict[str, str] = {
    "order_confirmed": "Your order #{order_id} has been confirmed. Total: ${total}.",
    "order_cancelled": "Your order #{order_id} has been cancelled.",
    "refund_issued": "A refund of ${amount} has been issued for order #{order_id}.",
    "shipment_dispatched": "Your order #{order_id} has been shipped. ETA: {eta}.",
    "password_reset": "Use code {code} to reset your password. Expires in {expiry_minutes} minutes.",
}


def render_template(event_type: str, context: dict[str, Any]) -> str:
    """Render a notification template with the given context.

    Parameters
    ----------
    event_type:
        The event type key (must exist in the template registry).
    context:
        Variable substitutions for the template string.

    Returns
    -------
    str
        The rendered message body.

    Raises
    ------
    KeyError
        If no template is registered for *event_type*.
    """
    template = _TEMPLATES[event_type]
    rendered = template.format(**context)
    logger.debug("Rendered template for '%s': %s", event_type, rendered[:80])
    return rendered
