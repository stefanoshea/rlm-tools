"""In-process background task queue.

Provides a simple FIFO queue for scheduling and executing background
jobs.  Tasks are registered by name and dispatched via
:meth:`TaskQueue.process_next`.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class TaskPayload:
    """Serialisable task descriptor."""

    task_name: str
    payload: dict[str, Any] = field(default_factory=dict)


class TaskQueue:
    """FIFO queue backed by an in-memory deque.

    Register callables via :meth:`register`, then enqueue work with
    :meth:`enqueue` and drain with :meth:`process_next`.
    """

    def __init__(self) -> None:
        self._queue: deque[TaskPayload] = deque()
        self._task_registry: dict[str, Callable[..., Any]] = {}

    def register(self, task_name: str, handler: Callable[..., Any]) -> None:
        """Register a handler for a named task."""
        self._task_registry[task_name] = handler

    def enqueue(self, task_name: str, payload: dict[str, Any] | None = None) -> None:
        """Add a task to the end of the queue."""
        if task_name not in self._task_registry:
            raise ValueError(f"Unknown task: {task_name}")
        self._queue.append(TaskPayload(task_name=task_name, payload=payload or {}))
        logger.debug("Enqueued task %s", task_name)

    def process_next(self) -> Any | None:
        """Pop and execute the next task. Returns the handler result."""
        if not self._queue:
            return None
        item = self._queue.popleft()
        handler = self._task_registry[item.task_name]
        logger.info("Processing task %s", item.task_name)
        return handler(**item.payload)
