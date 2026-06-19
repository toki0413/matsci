"""Celery-backed task backend.

Requires ``celery`` and a broker (e.g. Redis). Install with
``pip install celery[redis]``.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from huginn.queue.base import TaskBackend, TaskResult


class CeleryTaskBackend(TaskBackend):
    """Celery task backend.

    The actual Celery app and worker configuration live outside this class;
    this backend only submits tasks and polls results via the Celery API.
    """

    def __init__(self, app: Any | None = None) -> None:
        self._app = app

    def _get_app(self) -> Any:
        if self._app is None:
            try:
                from celery import Celery
            except ImportError as err:
                raise ImportError(
                    "CeleryTaskBackend requires 'celery'. "
                    "Install with: pip install celery[redis]"
                ) from err

            broker = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
            backend = os.environ.get("CELERY_RESULT_BACKEND", broker)
            self._app = Celery("huginn", broker=broker, backend=backend)
        return self._app

    def register_task(
        self,
        name: str,
        handler: Callable[..., Any],
    ) -> None:
        app = self._get_app()
        # In Celery, tasks are registered by decorating functions. We wrap the
        # handler in a Celery task dynamically.
        app.task(name=name)(handler)

    def send_task(
        self,
        name: str,
        args: tuple[Any, ...] | None = None,
        kwargs: dict[str, Any] | None = None,
        task_id: str | None = None,
    ) -> str:
        app = self._get_app()
        result = app.send_task(
            name,
            args=args,
            kwargs=kwargs,
            task_id=task_id,
        )
        return result.id

    def get_result(self, task_id: str) -> TaskResult:
        app = self._get_app()
        async_result = app.AsyncResult(task_id)
        status = async_result.status
        if status == "SUCCESS":
            return TaskResult(
                task_id=task_id,
                status="SUCCESS",
                result=async_result.result,
            )
        if status == "FAILURE":
            return TaskResult(
                task_id=task_id,
                status="FAILURE",
                error=str(async_result.result),
            )
        return TaskResult(task_id=task_id, status=status)

    def wait_for(
        self,
        task_id: str,
        timeout: float | None = None,
        poll_interval: float = 1.0,
    ) -> TaskResult:
        app = self._get_app()
        async_result = app.AsyncResult(task_id)
        try:
            value = async_result.get(timeout=timeout, interval=poll_interval)
            return TaskResult(
                task_id=task_id,
                status="SUCCESS",
                result=value,
            )
        except Exception as exc:
            return TaskResult(
                task_id=task_id,
                status="FAILURE",
                error=str(exc),
            )
