"""Notification dispatch engine.

Routes events to the appropriate notification channel based on the
application's ``NOTIFICATION_CHANNEL_MAP`` configuration.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from app.settings import NOTIFICATION_CHANNEL_MAP

logger = logging.getLogger(__name__)

DEFAULT_CHANNEL: str = "email"


class Channel(Protocol):
    """Protocol that all notification channels must satisfy."""

    def send(self, payload: dict[str, Any]) -> None: ...


# ---------------------------------------------------------------------------
# Channel registry — lazily populated on first dispatch
# ---------------------------------------------------------------------------
_CHANNEL_REGISTRY: dict[str, type] = {}
_registry_loaded: bool = False


def _load_registry() -> None:
    """Import and register the built-in channel classes."""
    global _registry_loaded
    if _registry_loaded:
        return

    from app.notifications.channels.email_channel import EmailChannel
    from app.notifications.channels.sms_channel import SmsChannel
    from app.notifications.channels.webhook_channel import WebhookChannel

    _CHANNEL_REGISTRY["email"] = EmailChannel
    _CHANNEL_REGISTRY["sms"] = SmsChannel
    _CHANNEL_REGISTRY["webhook"] = WebhookChannel
    _registry_loaded = True


def dispatch(event_type: str, payload: dict[str, Any]) -> None:
    """Route a notification event to the configured channel.

    Falls back to :data:`DEFAULT_CHANNEL` when the event type is not
    explicitly mapped in settings.
    """
    _load_registry()

    channel_name = NOTIFICATION_CHANNEL_MAP.get(event_type, DEFAULT_CHANNEL)
    channel_cls = _CHANNEL_REGISTRY.get(channel_name)

    if channel_cls is None:
        logger.error("No channel registered for '%s' — dropping event '%s'", channel_name, event_type)
        return

    logger.info("Dispatching '%s' via channel '%s'", event_type, channel_name)
    channel_instance = channel_cls()
    channel_instance.send({**payload, "_event": event_type})
