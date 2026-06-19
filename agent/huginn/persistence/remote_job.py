"""Remote job backend abstraction and implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from huginn.execution.remote_job_store import RemoteJobRecord, RemoteJobStore


class RemoteJobBackend(ABC):
    """Abstract backend for persisting remote HPC job records."""

    @abstractmethod
    def load(self) -> list[RemoteJobRecord]:
        """Load all stored job records."""
        raise NotImplementedError

    @abstractmethod
    def save(self, records: list[RemoteJobRecord]) -> None:
        """Persist all job records."""
        raise NotImplementedError

    @abstractmethod
    def add_or_update(self, record: RemoteJobRecord) -> None:
        """Insert or update a single record."""
        raise NotImplementedError

    @abstractmethod
    def get(self, local_id: str) -> RemoteJobRecord | None:
        """Return a record by local ID."""
        raise NotImplementedError

    @abstractmethod
    def list_jobs(self) -> list[RemoteJobRecord]:
        """Return all records, newest first."""
        raise NotImplementedError


class NullRemoteJobBackend(RemoteJobBackend):
    """No-op remote job backend for tests and ephemeral executors."""

    def load(self) -> list[RemoteJobRecord]:
        return []

    def save(self, records: list[RemoteJobRecord]) -> None:
        return None

    def add_or_update(self, record: RemoteJobRecord) -> None:
        return None

    def get(self, local_id: str) -> RemoteJobRecord | None:
        return None

    def list_jobs(self) -> list[RemoteJobRecord]:
        return []


class JSONRemoteJobBackend(RemoteJobBackend):
    """JSON-file-backed remote job backend.

    Wraps the existing ``RemoteJobStore`` implementation so it can be injected
    as a backend port.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        workspace: str | Path = ".",
        max_records: int = 1000,
    ) -> None:
        self._impl = RemoteJobStore(
            path=path,
            workspace=workspace,
            max_records=max_records,
        )

    def load(self) -> list[RemoteJobRecord]:
        return self._impl.load()

    def save(self, records: list[RemoteJobRecord]) -> None:
        self._impl.save(records)

    def add_or_update(self, record: RemoteJobRecord) -> None:
        self._impl.add_or_update(record)

    def get(self, local_id: str) -> RemoteJobRecord | None:
        return self._impl.get(local_id)

    def list_jobs(self) -> list[RemoteJobRecord]:
        return self._impl.list_jobs()
