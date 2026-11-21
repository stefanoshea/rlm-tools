"""SMS notification channel.

Sends notifications via an HTTP-based SMS provider API.
"""

from __future__ import annotations

import logging
from typing import Any

from app.settings import get_setting

logger = logging.getLogger(__name__)

_SMS_API_URL: str = get_setting("notifications.sms.api_url", "https://sms.provider.io/v1/send")
_SMS_API_KEY: str = get_setting("notifications.sms.api_key", "")
_SMS_FROM_NUMBER: str = get_setting("notifications.sms.from_number", "+10000000000")


class SmsChannel:
    """Delivers notifications as SMS messages via an HTTP adapter."""

    def __init__(self) -> None:
        self._api_url = _SMS_API_URL
        self._api_key = _SMS_API_KEY
        self._from_number = _SMS_FROM_NUMBER

    def send(self, payload: dict[str, Any]) -> None:
        """Send an SMS notification.

        Expects ``phone_number`` in the payload for the recipient.
        """
        phone = payload.get("phone_number", "")
        event = payload.get("_event", "unknown")
        logger.info(
            "Sending SMS: from=%s to=%s event=%s",
            self._from_number, phone, event,
        )
        # HTTP POST to SMS provider would happen here
