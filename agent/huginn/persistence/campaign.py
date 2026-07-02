"""Campaign store — persistence backend for the tool scheduler and research campaigns.

This file ships the ``jobs`` table portion of the planned CampaignStore (Tier 1 P6).
The ``campaigns`` / ``checkpoints`` / ``provenance`` tables are added later by P3;
the scheduler only needs ``jobs`` to do admission, queueing and cross-process recovery.

Follows the ABC + Null + impl pattern of ``persistence/remote_job.py``.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class JobRecord:
    """One row in the ``jobs`` table — a unit of compute tracked by the scheduler."""

    job_id: str
    tool_name: str
    status: str  # queued | running | finished | failed | orphaned
    cost_tier: str = "none"  # heavy | light | none
    campaign_id: str | None = None
    working_dir: str | None = None
    compute_action: str | None = None
    cores_requested: float | None = None
    queue_position: int | None = None
    admitted_at: float | None = None
    started_at: float | None = None
    finished_at: float | None = None
    result_json: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "tool_name": self.tool_name,
            "status": self.status,
            "cost_tier": self.cost_tier,
            "campaign_id": self.campaign_id,
            "working_dir": self.working_dir,
            "compute_action": self.compute_action,
            "cores_requested": self.cores_requested,
            "queue_position": self.queue_position,
            "admitted_at": self.admitted_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "result_json": self.result_json,
            "error": self.error,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> JobRecord:
        return cls(
            job_id=row["job_id"],
            tool_name=row["tool_name"],
            status=row["status"],
            cost_tier=row["cost_tier"] or "none",
            campaign_id=row["campaign_id"],
            working_dir=row["working_dir"],
            compute_action=row["compute_action"],
            cores_requested=row["cores_requested"],
            queue_position=row["queue_position"],
            admitted_at=row["admitted_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            result_json=row["result_json"],
            error=row["error"],
        )


class CampaignStoreBackend(ABC):
    """Abstract backend for scheduler job records.

    The scheduler calls these methods to persist async-job lifecycle so that a
    crash mid-run leaves enough state on disk to mark the job orphaned on restart.
    """

    @abstractmethod
    def upsert_job(self, record: JobRecord) -> None:
        """Insert or update a job row by job_id."""
        raise NotImplementedError

    @abstractmethod
    def get_job(self, job_id: str) -> JobRecord | None:
        raise NotImplementedError

    @abstractmethod
    def list_jobs_by_status(self, status: str) -> list[JobRecord]:
        raise NotImplementedError

    @abstractmethod
    def list_queued_jobs(self) -> list[JobRecord]:
        """Queued jobs ordered by queue_position ascending (FIFO)."""
        raise NotImplementedError

    @abstractmethod
    def claim_next_queued(self) -> JobRecord | None:
        """Atomically pop the head of the queue (lowest queue_position).

        Returns the claimed record (now to be marked running by the caller) or
        None if the queue is empty.
        """
        raise NotImplementedError

    @abstractmethod
    def next_queue_position(self) -> int:
        """Return the next queue_position value (monotonic within the queue)."""
        raise NotImplementedError


class NullCampaignStore(CampaignStoreBackend):
    """In-memory no-op store — used when P3's SQLite store is not yet wired.

    Keeps enough state for the scheduler's in-memory queueing and the unit tests
    that do not exercise cross-process recovery.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()
        self._pos = 0

    def upsert_job(self, record: JobRecord) -> None:
        with self._lock:
            if record.queue_position is None and record.status == "queued":
                record.queue_position = self._pos
                self._pos += 1
            self._jobs[record.job_id] = record

    def get_job(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs_by_status(self, status: str) -> list[JobRecord]:
        with self._lock:
            return [r for r in self._jobs.values() if r.status == status]

    def list_queued_jobs(self) -> list[JobRecord]:
        with self._lock:
            queued = [r for r in self._jobs.values() if r.status == "queued"]
            queued.sort(key=lambda r: r.queue_position if r.queue_position is not None else 0)
            return queued

    def claim_next_queued(self) -> JobRecord | None:
        with self._lock:
            queued = [r for r in self._jobs.values() if r.status == "queued"]
            if not queued:
                return None
            queued.sort(key=lambda r: r.queue_position if r.queue_position is not None else 0)
            head = queued[0]
            return head

    def next_queue_position(self) -> int:
        with self._lock:
            self._pos += 1
            return self._pos


class SqliteCampaignStore(CampaignStoreBackend):
    """SQLite-backed job store. Default path: ``<workspace>/.huginn/campaigns.sqlite``.

    The ``jobs`` table is created on first use. P3 will add the campaigns /
    checkpoints / provenance tables to this same database.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None

    def _resolve_path(self) -> Path:
        import os

        if self.path is not None:
            return self.path
        base = os.environ.get("HUGINN_CACHE_DIR")
        if base:
            return Path(base) / "campaigns.sqlite"
        return Path(".huginn") / "campaigns.sqlite"

    def _connect(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        resolved = self._resolve_path()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(resolved), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                tool_name TEXT NOT NULL,
                status TEXT NOT NULL,
                cost_tier TEXT NOT NULL DEFAULT 'none',
                campaign_id TEXT,
                working_dir TEXT,
                compute_action TEXT,
                cores_requested REAL,
                queue_position INTEGER,
                admitted_at REAL,
                started_at REAL,
                finished_at REAL,
                result_json TEXT,
                error TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
            CREATE INDEX IF NOT EXISTS idx_jobs_campaign ON jobs(campaign_id);
            """
        )
        conn.commit()
        self._conn = conn
        return conn

    def upsert_job(self, record: JobRecord) -> None:
        cols = [
            "job_id", "tool_name", "status", "cost_tier", "campaign_id",
            "working_dir", "compute_action", "cores_requested", "queue_position",
            "admitted_at", "started_at", "finished_at", "result_json", "error",
        ]
        values = (
            record.job_id, record.tool_name, record.status, record.cost_tier,
            record.campaign_id, record.working_dir, record.compute_action,
            record.cores_requested, record.queue_position, record.admitted_at,
            record.started_at, record.finished_at, record.result_json, record.error,
        )
        placeholders = ",".join(["?"] * len(cols))
        col_list = ",".join(cols)
        # Update every column except the primary key on conflict.
        update_list = ",".join(f"{c}=excluded.{c}" for c in cols if c != "job_id")
        sql = (
            f"INSERT INTO jobs ({col_list}) VALUES ({placeholders}) "
            f"ON CONFLICT(job_id) DO UPDATE SET {update_list}"
        )
        with self._lock:
            conn = self._connect()
            conn.execute(sql, values)
            conn.commit()

    def get_job(self, job_id: str) -> JobRecord | None:
        with self._lock:
            conn = self._connect()
            row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            return JobRecord.from_row(row) if row is not None else None

    def list_jobs_by_status(self, status: str) -> list[JobRecord]:
        with self._lock:
            conn = self._connect()
            rows = conn.execute(
                "SELECT * FROM jobs WHERE status=? ORDER BY started_at", (status,)
            ).fetchall()
            return [JobRecord.from_row(r) for r in rows]

    def list_queued_jobs(self) -> list[JobRecord]:
        with self._lock:
            conn = self._connect()
            rows = conn.execute(
                "SELECT * FROM jobs WHERE status='queued' ORDER BY queue_position"
            ).fetchall()
            return [JobRecord.from_row(r) for r in rows]

    def claim_next_queued(self) -> JobRecord | None:
        with self._lock:
            conn = self._connect()
            row = conn.execute(
                "SELECT * FROM jobs WHERE status='queued' ORDER BY queue_position LIMIT 1"
            ).fetchone()
            return JobRecord.from_row(row) if row is not None else None

    def next_queue_position(self) -> int:
        with self._lock:
            conn = self._connect()
            row = conn.execute(
                "SELECT COALESCE(MAX(queue_position), -1) AS m FROM jobs WHERE status='queued'"
            ).fetchone()
            return int(row["m"]) + 1

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
