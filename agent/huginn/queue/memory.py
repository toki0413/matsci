"""In-memory task backend for development and tests.

Runs tasks synchronously in the calling process. Not suitable for production
but requires no external broker.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from typing import Any

from huginn.queue.base import TaskBackend, TaskResult


class InMemoryTaskBackend(TaskBackend):
    """Synchronous in-memory task backend.

    Tasks are stored as pending when submitted and executed lazily when
    ``wait_for``/``get_result`` is called. This mirrors the behaviour of a real
    broker (Celery/RQ) where ``send_task`` only enqueues work and completion is
    polled later. Handlers must be synchronous; long-running handlers can be
    awaited from async code via ``asyncio.to_thread``.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, Callable[..., Any]] = {}
        self._pending: dict[str, tuple[str, tuple[Any, ...], dict[str, Any]]] = {}
        self._results: dict[str, TaskResult] = {}
        self._lock = threading.Lock()

    def register_task(
        self,
        name: str,
        handler: Callable[..., Any],
    ) -> None:
        self._handlers[name] = handler

    def send_task(
        self,
        name: str,
        args: tuple[Any, ...] | None = None,
        kwargs: dict[str, Any] | None = None,
        task_id: str | None = None,
    ) -> str:
        args = args or ()
        kwargs = kwargs or {}
        task_id = task_id or uuid.uuid4().hex
        with self._lock:
            self._pending[task_id] = (name, args, kwargs)
        return task_id

    def _run_pending(self, task_id: str) -> TaskResult:
        with self._lock:
            if task_id in self._results:
                return self._results[task_id]
            pending = self._pending.pop(task_id, None)

        if pending is None:
            return TaskResult(
                task_id=task_id,
                status="FAILURE",
                error=f"Unknown task_id: {task_id}",
            )

        name, args, kwargs = pending
        handler = self._handlers.get(name)
        if handler is None:
            result = TaskResult(
                task_id=task_id,
                status="FAILURE",
                error=f"No handler registered for task '{name}'",
            )
        else:
            try:
                value = handler(*args, **kwargs)
                result = TaskResult(
                    task_id=task_id,
                    status="SUCCESS",
                    result=value,
                )
            except Exception as exc:
                result = TaskResult(
                    task_id=task_id,
                    status="FAILURE",
                    error=str(exc),
                )

        with self._lock:
            self._results[task_id] = result
        return result

    def get_result(self, task_id: str) -> TaskResult:
        with self._lock:
            if task_id in self._pending:
                return TaskResult(
                    task_id=task_id,
                    status="PENDING",
                )
            if task_id in self._results:
                return self._results[task_id]
        return self._run_pending(task_id)

    def wait_for(
        self,
        task_id: str,
        timeout: float | None = None,
        poll_interval: float = 1.0,
    ) -> TaskResult:
        return self._run_pending(task_id)
