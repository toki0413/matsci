"""Persistent store for remote job records.

Jobs submitted via ``RemoteExecutor`` are written to disk so users can list,
query, and cancel them across CLI invocations and agent restarts.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RemoteJobRecord:
    """Tracked record of a submitted remote job."""

    local_id: str
    scheduler_id: str
    command: list[str]
    cwd: str
    queue: str | None = None
    status: str = "PENDING"
    exit_code: int | None = None
    submitted_at: float = field(default_factory=lambda: 0.0)
    completed_at: float | None = None
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize record to a JSON-friendly dict."""
        return {
            "local_id": self.local_id,
            "scheduler_id": self.scheduler_id,
            "command": self.command,
            "cwd": self.cwd,
            "queue": self.queue,
            "status": self.status,
            "exit_code": self.exit_code,
            "submitted_at": self.submitted_at,
            "completed_at": self.completed_at,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RemoteJobRecord:
        """Restore a record from a serialized dict."""
        return cls(
            local_id=data["local_id"],
            scheduler_id=data["scheduler_id"],
            command=list(data.get("command", [])),
            cwd=data.get("cwd", ""),
            queue=data.get("queue"),
            status=data.get("status", "PENDING"),
            exit_code=data.get("exit_code"),
            submitted_at=data.get("submitted_at", 0.0),
            completed_at=data.get("completed_at"),
            message=data.get("message"),
        )


class RemoteJobStore:
    """JSON-backed persistent store for remote job records.

    The default location is ``<workspace>/.huginn/remote_jobs.json``.
    To prevent unbounded growth, the store automatically prunes the oldest
    terminal records when ``max_records`` is exceeded.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        workspace: str | Path = ".",
        max_records: int = 1000,
    ):
        if path is not None:
            self.path = Path(path).expanduser().resolve()
        else:
            self.path = (
                Path(workspace).expanduser().resolve() / ".huginn" / "remote_jobs.json"
            )
        self.max_records = max_records
        self._lock = threading.Lock()
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> list[RemoteJobRecord]:
        """Load all records from disk."""
        if not self.path.exists():
            return []
        try:
            with self._lock, self.path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                return []
            return [
                RemoteJobRecord.from_dict(item)
                for item in data
                if isinstance(item, dict)
            ]
        except Exception as exc:
            logger.warning("Failed to load remote job store %s: %s", self.path, exc)
            return []

    def save(self, records: list[RemoteJobRecord]) -> None:
        """Persist records to disk atomically."""
        self._ensure_dir()
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            with self._lock, tmp.open("w", encoding="utf-8") as f:
                json.dump([r.to_dict() for r in records], f, indent=2, default=str)
            tmp.replace(self.path)
        except Exception as exc:
            logger.warning("Failed to save remote job store %s: %s", self.path, exc)

    def list_jobs(self) -> list[RemoteJobRecord]:
        """Return all stored records, newest first."""
        return sorted(self.load(), key=lambda j: j.submitted_at, reverse=True)

    def get(self, local_id: str) -> RemoteJobRecord | None:
        """Return a record by local ID."""
        for record in self.load():
            if record.local_id == local_id:
                return record
        return None

    def add_or_update(self, record: RemoteJobRecord) -> None:
        """Insert a new record or overwrite an existing one with the same local_id."""
        records = self.load()
        by_id = {r.local_id: r for r in records}
        by_id[record.local_id] = record
        records = self._cap_records(list(by_id.values()), max_records=self.max_records)
        self.save(records)

    def _cap_records(
        self,
        records: list[RemoteJobRecord],
        max_records: int,
    ) -> list[RemoteJobRecord]:
        """Drop oldest terminal records if the store exceeds max_records."""
        if len(records) <= max_records:
            return records
        terminal = {"COMPLETED", "FAILED", "CANCELLED"}
        terminal_records = [r for r in records if r.status in terminal]
        keep_non_terminal = [r for r in records if r.status not in terminal]
        # Keep newest terminal records; non-terminal records are never dropped.
        terminal_records.sort(key=lambda r: r.submitted_at, reverse=True)
        keep_terminal = terminal_records[: max(0, max_records - len(keep_non_terminal))]
        return keep_non_terminal + keep_terminal

    def prune(
        self,
        max_records: int | None = None,
        keep_statuses: set[str] | None = None,
    ) -> int:
        """Prune records down to ``max_records`` (defaults to self.max_records).

        Only prunes terminal statuses unless ``keep_statuses`` restricts further.
        Returns number of removed records.
        """
        target = max_records if max_records is not None else self.max_records
        records = self.load()
        if len(records) <= target:
            return 0
        terminal = keep_statuses or {"COMPLETED", "FAILED", "CANCELLED"}
        pruned = self._cap_records(
            [r for r in records if r.status in terminal]
            + [r for r in records if r.status not in terminal],
            max_records=target,
        )
        removed = len(records) - len(pruned)
        if removed:
            self.save(pruned)
        return removed

    def remove(self, local_id: str) -> bool:
        """Remove a record from the store."""
        records = self.load()
        new_records = [r for r in records if r.local_id != local_id]
        if len(new_records) == len(records):
            return False
        self.save(new_records)
        return True
