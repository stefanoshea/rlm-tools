# TODO: not wired up yet, see JIRA-4521
"""Slack notification channel (not yet active).

This channel posts to a Slack webhook URL but is **not registered**
in the dispatcher channel registry.  See JIRA-4521 for the
integration ticket.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.settings import get_setting

logger = logging.getLogger(__name__)

_SLACK_WEBHOOK_URL: str = get_setting("notifications.slack.webhook_url", "")


class SlackChannel:
    """Posts notifications to a Slack incoming webhook."""

    def __init__(self) -> None:
        self._url = _SLACK_WEBHOOK_URL

    def send(self, payload: dict[str, Any]) -> None:
        """Send a Slack message.  Not currently routed by the dispatcher."""
        event = payload.get("_event", "unknown")
        if not self._url:
            logger.warning("Slack webhook URL not configured")
            return
        body = json.dumps({"text": f"[{event}] {json.dumps(payload)}"})
        logger.info("Slack notification: url=%s event=%s", self._url, event)
