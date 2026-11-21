"""Background worker for notification dispatch.

Processes notification events asynchronously so that the main request
path is never blocked by email or SMS delivery.
"""

from __future__ import annotations

import logging
from typing import Any

from app.notifications.dispatcher import NotificationDispatcher

logger = logging.getLogger(__name__)


class NotificationWorker:
    """Consumes notification events and dispatches them.

    Uses :class:`~app.notifications.dispatcher.NotificationDispatcher`
    to route each event to the correct channel (email, SMS, webhook).
    """

    def __init__(self) -> None:
        self._dispatcher = NotificationDispatcher()

    def process_notification(self, event: dict[str, Any]) -> dict[str, Any]:
        """Handle a single notification event.

        Parameters
        ----------
        event:
            Dictionary with ``type``, ``recipient``, and ``payload`` keys.

        Returns
        -------
        dict[str, Any]
            Delivery result including channel used and success flag.
        """
        event_type: str = event.get("type", "unknown")
        recipient: str = event.get("recipient", "")
        payload: dict[str, Any] = event.get("payload", {})

        logger.info("Processing notification: type=%s recipient=%s", event_type, recipient)
        result = self._dispatcher.dispatch(
            event_type=event_type,
            recipient=recipient,
            payload=payload,
        )
        return {"event_type": event_type, "delivered": True, "channel": result}
