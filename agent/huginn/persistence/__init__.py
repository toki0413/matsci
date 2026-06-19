"""Persistence backends for Huginn state.

Provides swappable backends for checkpoints, long-term memory, and remote job
records so the agent can run with SQLite/JSON in development and
Postgres/Redis/S3 in production.
"""

from __future__ import annotations

from huginn.persistence.checkpointer import (
    CheckpointerBackend,
    SQLiteCheckpointerBackend,
)
from huginn.persistence.memory import MemoryBackend, SQLiteMemoryBackend
from huginn.persistence.remote_job import (
    JSONRemoteJobBackend,
    NullRemoteJobBackend,
    RemoteJobBackend,
)

__all__ = [
    "CheckpointerBackend",
    "SQLiteCheckpointerBackend",
    "MemoryBackend",
    "SQLiteMemoryBackend",
    "RemoteJobBackend",
    "JSONRemoteJobBackend",
    "NullRemoteJobBackend",
]
