"""In-memory task backend for development and tests.

Runs tasks synchronously in the calling process. Not suitable for production
but requires no external broker.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable
from typing import Any

from huginn.queue.base import TaskBackend, TaskResult

_log = logging.getLogger("huginn.queue.memory")


class InMemoryTaskBackend(TaskBackend):
    """Synchronous in-memory task backend.

    Tasks are stored as pending when submitted and executed lazily when
    ``wait_for``/``get_result`` is called. This mirrors the behaviour of a real
    broker (Celery/RQ) where ``send_task`` only enqueues work and completion is
    polled later. Handlers must be synchronous; long-running handlers can be
    awaited from async code via ``asyncio.to_thread``.

    When a *dlq* (dead letter queue) is wired in, tasks that fail are copied
    there instead of being silently dropped — useful for inspecting/replaying
    failures. There is no automatic retry yet; once retries are added, the DLQ
    enqueue should happen only after the retry budget is exhausted.
    """

    def __init__(self, *, dlq: Any | None = None) -> None:
        self._handlers: dict[str, Callable[..., Any]] = {}
        self._pending: dict[str, tuple[str, tuple[Any, ...], dict[str, Any]]] = {}
        self._results: dict[str, TaskResult] = {}
        self._lock = threading.Lock()
        self._dlq = dlq

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

    def _maybe_dlq(
        self,
        task_id: str,
        name: str | None,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        reason: str,
    ) -> None:
        """Best-effort enqueue of a failed task to the dead letter queue.

        Swallows DLQ errors so a failing DLQ can never mask the original
        task failure — the task result already carries the error.
        """
        if self._dlq is None:
            return
        try:
            self._dlq.enqueue(
                task_id,
                {"name": name, "args": list(args), "kwargs": kwargs},
                reason,
                task_name=name,
            )
        except Exception as dlq_err:  # noqa: BLE001
            _log.error(
                "Failed to enqueue task %s to DLQ: %s", task_id, dlq_err
            )

    def _run_pending(self, task_id: str) -> TaskResult:
        with self._lock:
            if task_id in self._results:
                return self._results[task_id]
            pending = self._pending.pop(task_id, None)

        if pending is None:
            result = TaskResult(
                task_id=task_id,
                status="FAILURE",
                error=f"Unknown task_id: {task_id}",
            )
            self._maybe_dlq(task_id, None, (), {}, result.error)
        else:
            name, args, kwargs = pending
            handler = self._handlers.get(name)
            if handler is None:
                result = TaskResult(
                    task_id=task_id,
                    status="FAILURE",
                    error=f"No handler registered for task '{name}'",
                )
                self._maybe_dlq(task_id, name, args, kwargs, result.error)
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
                    # Permanent failure — park it in the DLQ for inspection.
                    self._maybe_dlq(task_id, name, args, kwargs, str(exc))

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
