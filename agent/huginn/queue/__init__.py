"""Task queue backends for Huginn."""

from __future__ import annotations

from huginn.queue.base import TaskBackend, TaskResult
from huginn.queue.memory import InMemoryTaskBackend

__all__ = [
    "TaskBackend",
    "TaskResult",
    "InMemoryTaskBackend",
]
