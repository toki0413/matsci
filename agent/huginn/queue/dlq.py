"""Dead letter queue — durable storage for tasks that exhausted all retries.

Failed tasks land here instead of being silently dropped, so operators can
inspect, replay, or discard them. Backed by SQLite in WAL mode so DLQ reads
never block the main queue's writes.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_log = logging.getLogger("huginn.queue.dlq")

# How long a dead letter sticks around before being auto-purged. 7 days is
# enough window for a human to notice and replay, without letting the table
# grow unbounded on a busy worker.
_DEFAULT_TTL_DAYS = 7.0


class DeadLetterQueue:
    """SQLite-backed dead letter queue with TTL-based auto-expiry."""

    def __init__(
        self,
        db_path: str | None = None,
        *,
        ttl_days: float = _DEFAULT_TTL_DAYS,
    ) -> None:
        if db_path is None:
            cache_dir = os.environ.get(
                "HUGINN_CACHE_DIR", str(Path.home() / ".huginn")
            )
            db_path = str(Path(cache_dir) / "dlq.db")
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl_days = ttl_days
        self._init_db()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        # WAL keeps reads from blocking while a failed task is being written.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS dead_letters (
                    dlq_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    task_name TEXT,
                    task_data TEXT NOT NULL,
                    failure_reason TEXT,
                    retry_count INTEGER DEFAULT 0,
                    failed_at REAL NOT NULL,
                    last_attempted_at REAL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_dlq_task_id ON dead_letters(task_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_dlq_failed_at ON dead_letters(failed_at)"
            )
            conn.commit()

    def _expire_old(self, conn: sqlite3.Connection) -> None:
        """Purge dead letters older than the TTL. Called on every mutation/read."""
        cutoff = time.time() - self.ttl_days * 86400.0
        conn.execute("DELETE FROM dead_letters WHERE failed_at < ?", (cutoff,))

    @staticmethod
    def _serialize(task_data: Any) -> str:
        """Normalize the task payload to a JSON string for storage."""
        if isinstance(task_data, (dict, list, tuple)):
            return json.dumps(task_data, default=str, ensure_ascii=False)
        if task_data is None:
            return "null"
        return str(task_data)

    def enqueue(
        self,
        task_id: str,
        task_data: Any,
        failure_reason: str,
        *,
        task_name: str | None = None,
    ) -> str:
        """Record a permanently failed task.

        If the task already has a dead letter, we bump its retry_count and
        refresh the failure reason instead of inserting a duplicate — this
        keeps the table readable when a flaky task fails repeatedly.
        """
        payload = self._serialize(task_data)
        now = time.time()
        with self._connect() as conn:
            self._expire_old(conn)
            existing = conn.execute(
                "SELECT dlq_id, retry_count FROM dead_letters "
                "WHERE task_id = ? ORDER BY failed_at DESC LIMIT 1",
                (task_id,),
            ).fetchone()
            if existing is not None:
                conn.execute(
                    "UPDATE dead_letters SET retry_count = retry_count + 1, "
                    "last_attempted_at = ?, failure_reason = ? WHERE dlq_id = ?",
                    (now, failure_reason, existing["dlq_id"]),
                )
                dlq_id = existing["dlq_id"]
            else:
                dlq_id = f"dlq_{uuid.uuid4().hex[:12]}"
                conn.execute(
                    """
                    INSERT INTO dead_letters
                    (dlq_id, task_id, task_name, task_data, failure_reason,
                     retry_count, failed_at, last_attempted_at)
                    VALUES (?, ?, ?, ?, ?, 0, ?, ?)
                    """,
                    (dlq_id, task_id, task_name, payload, failure_reason, now, now),
                )
            conn.commit()
        _log.warning(
            "Task %s enqueued to DLQ (%s): %s",
            task_id,
            dlq_id,
            failure_reason,
        )
        return dlq_id

    def dequeue(self) -> dict[str, Any] | None:
        """Peek the oldest dead letter (for manual retry inspection).

        Does NOT remove the entry — call ``requeue`` once you've resubmitted
        the task to the main queue.
        """
        with self._connect() as conn:
            self._expire_old(conn)
            row = conn.execute(
                "SELECT * FROM dead_letters ORDER BY failed_at ASC LIMIT 1"
            ).fetchone()
            return dict(row) if row is not None else None

    def list_failed(self, *, limit: int = 100) -> list[dict[str, Any]]:
        """List dead letters, newest first."""
        with self._connect() as conn:
            self._expire_old(conn)
            rows = conn.execute(
                "SELECT * FROM dead_letters ORDER BY failed_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def requeue(self, task_id: str) -> dict[str, Any] | None:
        """Remove a task from the DLQ so it can be resubmitted to the main queue.

        Returns the dead-letter row (carrying task_name/task_data) so the
        caller has what it needs to rebuild and resubmit the task. Returns
        None when the task_id isn't in the DLQ.
        """
        with self._connect() as conn:
            self._expire_old(conn)
            row = conn.execute(
                "SELECT * FROM dead_letters WHERE task_id = ? "
                "ORDER BY failed_at DESC LIMIT 1",
                (task_id,),
            ).fetchone()
            if row is None:
                return None
            conn.execute(
                "DELETE FROM dead_letters WHERE dlq_id = ?", (row["dlq_id"],)
            )
            conn.commit()
            return dict(row)

    def purge(self) -> int:
        """Drop all dead letters. Returns the count removed."""
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM dead_letters")
            conn.commit()
            return cur.rowcount

    def stats(self) -> dict[str, Any]:
        """Return basic counts for monitoring dashboards."""
        with self._connect() as conn:
            self._expire_old(conn)
            total = conn.execute(
                "SELECT COUNT(*) FROM dead_letters"
            ).fetchone()[0]
            oldest = conn.execute(
                "SELECT MIN(failed_at) FROM dead_letters"
            ).fetchone()[0]
        return {
            "total": total,
            "oldest_failed_at": oldest,
            "ttl_days": self.ttl_days,
        }
