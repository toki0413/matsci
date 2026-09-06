"""Task queue backend abstraction for Huginn.

Long-running computational work (HPC jobs, workflow stages) is submitted to a
backend instead of blocking the API event loop. Backends can be in-memory
(synchronous, for dev/tests), Celery, RQ, or any other task broker.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskResult:
    """Result of a queued task."""

    task_id: str
    status: str  # PENDING, RUNNING, SUCCESS, FAILURE
    result: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class TaskBackend(ABC):
    """Abstract task queue backend."""

    @abstractmethod
    def send_task(
        self,
        name: str,
        args: tuple[Any, ...] | None = None,
        kwargs: dict[str, Any] | None = None,
        task_id: str | None = None,
    ) -> str:
        """Submit a task and return its ID."""
        raise NotImplementedError

    @abstractmethod
    def get_result(self, task_id: str) -> TaskResult:
        """Return the current result/status for ``task_id``."""
        raise NotImplementedError

    @abstractmethod
    def wait_for(
        self,
        task_id: str,
        timeout: float | None = None,
        poll_interval: float = 1.0,
    ) -> TaskResult:
        """Block until the task reaches a terminal state or times out."""
        raise NotImplementedError

    @abstractmethod
    def register_task(
        self,
        name: str,
        handler: Callable[..., Any],
    ) -> None:
        """Register a handler for ``name``.

        In-memory backends use this directly. Celery/RQ backends discover
        workers via module imports.
        """
        raise NotImplementedError
