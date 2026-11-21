"""Audit logging for sensitive administrative actions.

Every mutation performed through the admin interface is recorded so that
the security team can review changes after the fact.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from app.adapters.database import get_connection

logger = logging.getLogger(__name__)


@dataclass
class AuditEntry:
    """Single audit trail record."""

    user: str
    action: str
    resource: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


class AuditLogger:
    """Persists audit entries to the database.

    All writes go through the shared database adapter so they participate
    in the same connection pool as the rest of the application.
    """

    TABLE = "audit_log"

    def log_action(
        self,
        user: str,
        action: str,
        resource: str,
        metadata: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """Record an administrative action.

        Parameters
        ----------
        user:
            Username or service account that performed the action.
        action:
            Short verb describing what happened (e.g. ``"delete"``).
        resource:
            Identifier of the affected resource.
        metadata:
            Optional extra context stored as JSON.
        """
        entry = AuditEntry(
            user=user,
            action=action,
            resource=resource,
            metadata=metadata or {},
        )
        conn = get_connection()
        conn.execute(
            f"INSERT INTO {self.TABLE} (user, action, resource, ts, meta) "
            "VALUES (?, ?, ?, ?, ?)",
            (entry.user, entry.action, entry.resource, entry.timestamp, entry.metadata),
        )
        logger.info("Audit: %s performed %s on %s", user, action, resource)
        return entry
