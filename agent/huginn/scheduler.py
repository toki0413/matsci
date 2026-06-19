"""Lightweight file-based task scheduler for Huginn.

Stores scheduled jobs in a workspace JSON file. The ``start`` command runs a
blocking daemon that wakes up every minute, checks for due jobs using cron
expressions, and executes them as subprocess commands.
"""

from __future__ import annotations

import re
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class ScheduledJob:
    """A single scheduled command."""

    id: str
    cron: str
    command: str
    enabled: bool = True
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    last_run: str | None = None
    run_count: int = 0


class ScheduleManager:
    """Manage and execute scheduled jobs for a workspace."""

    _CRON_RE = re.compile(r"^\S+\s+\S+\s+\S+\s+\S+\s+\S+$")

    def __init__(self, workspace: str | Path):
        self.workspace = Path(workspace).resolve()
        self._path = self.workspace / ".huginn_schedule.json"

    def add(self, cron: str, command: str) -> str:
        """Add a new scheduled job and return its ID."""
        cron = cron.strip()
        if not self._CRON_RE.match(cron):
            raise ValueError(
                "Invalid cron expression; expected 5 fields: "
                "minute hour day-of-month month day-of-week"
            )
        if not command.strip():
            raise ValueError("Command cannot be empty")

        job = ScheduledJob(
            id=uuid.uuid4().hex[:12],
            cron=cron,
            command=command.strip(),
        )
        jobs = self._load()
        jobs.append(job)
        self._save(jobs)
        return job.id

    def list(self) -> list[ScheduledJob]:
        """Return all scheduled jobs."""
        return self._load()

    def remove(self, job_id: str) -> bool:
        """Remove a scheduled job by ID."""
        jobs = self._load()
        kept = [j for j in jobs if j.id != job_id]
        if len(kept) == len(jobs):
            return False
        self._save(kept)
        return True

    def enable(self, job_id: str, enabled: bool = True) -> bool:
        """Enable or disable a scheduled job."""
        jobs = self._load()
        found = False
        for job in jobs:
            if job.id == job_id:
                job.enabled = enabled
                found = True
        if not found:
            return False
        self._save(jobs)
        return True

    def due_jobs(self, now: datetime | None = None) -> list[ScheduledJob]:
        """Return jobs whose cron expression matches the current minute."""
        now = now or datetime.now(UTC)
        try:
            from croniter import croniter
        except ImportError as exc:
            raise RuntimeError(
                "Cron parsing requires croniter. Install: pip install croniter"
            ) from exc

        jobs = self._load()
        due: list[ScheduledJob] = []
        for job in jobs:
            if not job.enabled:
                continue
            if croniter.match(job.cron, now):
                due.append(job)
        return due

    def run_due(self, now: datetime | None = None) -> list[dict[str, Any]]:
        """Execute all due commands and record their run time."""
        due = self.due_jobs(now)
        results: list[dict[str, Any]] = []
        jobs = {j.id: j for j in self._load()}
        timestamp = datetime.now(UTC).isoformat()

        for job in due:
            start = time.time()
            try:
                proc = subprocess.run(
                    job.command,
                    shell=True,
                    cwd=self.workspace,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                ok = proc.returncode == 0
                result = {
                    "job_id": job.id,
                    "command": job.command,
                    "success": ok,
                    "returncode": proc.returncode,
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                    "duration": round(time.time() - start, 3),
                }
            except subprocess.TimeoutExpired:
                result = {
                    "job_id": job.id,
                    "command": job.command,
                    "success": False,
                    "error": "Timed out after 300s",
                }
            except Exception as exc:
                result = {
                    "job_id": job.id,
                    "command": job.command,
                    "success": False,
                    "error": str(exc),
                }

            if job.id in jobs:
                jobs[job.id].last_run = timestamp
                jobs[job.id].run_count += 1
            results.append(result)

        self._save(list(jobs.values()))
        return results

    def run_blocking(self, interval: int = 60) -> None:
        """Run the scheduler daemon until interrupted."""
        while True:
            self.run_due()
            time.sleep(interval)

    def _load(self) -> list[ScheduledJob]:
        if not self._path.exists():
            return []
        try:
            raw = self._path.read_text(encoding="utf-8")
            data = __import__("json").loads(raw)
        except Exception:
            return []
        return [ScheduledJob(**item) for item in data]

    def _save(self, jobs: list[ScheduledJob]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(job) for job in jobs]
        self._path.write_text(
            __import__("json").dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
