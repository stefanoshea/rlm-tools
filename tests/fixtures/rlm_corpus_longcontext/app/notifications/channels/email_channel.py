"""Email notification channel.

Sends notifications via SMTP using the application's mail configuration.
"""

from __future__ import annotations

import logging
from typing import Any

from app.settings import get_setting

logger = logging.getLogger(__name__)

_SMTP_HOST: str = get_setting("notifications.smtp.host", "localhost")
_SMTP_PORT: int = get_setting("notifications.smtp.port", 587)
_FROM_ADDRESS: str = get_setting("notifications.smtp.from_address", "no-reply@shipstation.io")


class EmailChannel:
    """Delivers notifications as email messages."""

    def __init__(self) -> None:
        self._host = _SMTP_HOST
        self._port = _SMTP_PORT
        self._from = _FROM_ADDRESS

    def send(self, payload: dict[str, Any]) -> None:
        """Send an email notification.

        The payload must include a ``recipient`` key with the target
        email address.
        """
        recipient = payload.get("recipient", "unknown@example.com")
        event = payload.get("_event", "unknown")
        logger.info(
            "Sending email: from=%s to=%s event=%s host=%s:%d",
            self._from, recipient, event, self._host, self._port,
        )
        # Actual SMTP send would happen here
