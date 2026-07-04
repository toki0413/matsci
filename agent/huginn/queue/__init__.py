"""Task queue backends for Huginn."""

from __future__ import annotations

from huginn.queue.base import TaskBackend, TaskResult
from huginn.queue.celery_backend import CeleryTaskBackend
from huginn.queue.dlq import DeadLetterQueue
from huginn.queue.memory import InMemoryTaskBackend

__all__ = [
    "TaskBackend",
    "TaskResult",
    "InMemoryTaskBackend",
    "CeleryTaskBackend",
    "DeadLetterQueue",
]
