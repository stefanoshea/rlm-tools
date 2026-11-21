"""Webhook notification channel.

Posts notification payloads to a configured external URL.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.settings import get_setting

logger = logging.getLogger(__name__)

_WEBHOOK_URL: str = get_setting("notifications.webhook.url", "")
_WEBHOOK_TIMEOUT: int = get_setting("notifications.webhook.timeout_seconds", 10)


class WebhookChannel:
    """Delivers notifications by POSTing JSON to an external webhook URL."""

    def __init__(self) -> None:
        self._url = _WEBHOOK_URL
        self._timeout = _WEBHOOK_TIMEOUT

    def send(self, payload: dict[str, Any]) -> None:
        """POST the notification payload as JSON to the webhook endpoint."""
        event = payload.get("_event", "unknown")
        if not self._url:
            logger.warning("Webhook URL not configured — dropping event '%s'", event)
            return

        body = json.dumps(payload)
        logger.info(
            "Posting webhook: url=%s event=%s timeout=%ds size=%d",
            self._url, event, self._timeout, len(body),
        )
        # HTTP POST would happen here
