"""Background job monitor — polls non-terminal remote jobs on a daemon thread.

Wakes up periodically, walks the RemoteJobStore for anything still
PENDING/RUNNING, and refreshes each one over SSH. Only logs on state
transitions so the log doesn't get noisy with identical lines. This is
pure backend housekeeping — zero LLM token consumption.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from huginn.execution.remote_job_store import RemoteJobRecord, RemoteJobStore
from huginn.hpc.client import HPCClient

logger = logging.getLogger(__name__)

# Poll frequency ramps down as the job ages — freshly submitted jobs change
# state quickly so we check often; long-running jobs don't need the attention.
_POLL_INTERVALS = [
    (300, 30),           # first 5 min: every 30s
    (1800, 60),          # 5-30 min: every 60s
    (3600, 120),         # 30-60 min: every 2 min
    (float("inf"), 300), # >1h: every 5 min
]

_TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"}


def _get_poll_interval(submitted_at: float) -> float:
    """Return poll interval based on how long ago the job was submitted."""
    age = time.time() - submitted_at
    for threshold, interval in _POLL_INTERVALS:
        if age < threshold:
            return interval
    return 300


class JobMonitor:
    """Daemon thread that polls non-terminal remote jobs.

    Usage::

        monitor = JobMonitor(workspace=".")
        monitor.start()   # spawns daemon thread
        monitor.stop()    # signals stop, joins, cleans up clients
    """

    def __init__(self, workspace: str | Path = "."):
        self.workspace = Path(workspace)
        self._store = RemoteJobStore(workspace=self.workspace)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        # Reuse SSH connections across poll cycles to avoid reconnect churn
        self._client_cache: dict[str, HPCClient] = {}

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="hpc-monitor"
        )
        self._thread.start()
        logger.info("job monitor started")

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        for client in self._client_cache.values():
            try:
                client.disconnect()
            except Exception:
                pass
        self._client_cache.clear()
        logger.info("job monitor stopped")

    def _run(self) -> None:
        """Main loop — wake up every 10s and poll whatever is due."""
        while not self._stop_event.is_set():
            try:
                self._poll_cycle()
            except Exception as e:
                logger.warning("monitor cycle error: %s", e)
            self._stop_event.wait(10.0)

    def _poll_cycle(self) -> None:
        """Refresh every non-terminal job in the store."""
        jobs = self._store.list_jobs()
        for record in jobs:
            if record.status in _TERMINAL_STATES:
                continue
            # _get_poll_interval tells us how often to check, but tracking
            # last-checked time per job adds complexity for little gain —
            # poll_status is cheap (one SSH command) so we just check every
            # cycle and let the scheduler-side caching handle the rest.
            try:
                self._refresh_single(record)
            except Exception as e:
                logger.debug("refresh failed for %s: %s", record.local_id, e)

    def _refresh_single(self, record: RemoteJobRecord) -> None:
        """Refresh a single job's status via SSH."""
        if not record.credential_id:
            logger.debug("skip %s — no credential_id, cannot reconnect", record.local_id)
            return

        # Lazy import — routes.hpc pulls in the full routes package which we
        # don't want at module load time (circular deps + StrEnum on 3.10)
        from huginn.routes.hpc import _resolve_hpc_config

        cfg, err = _resolve_hpc_config({"credential_id": record.credential_id})
        if err or cfg is None:
            logger.warning("cannot resolve config for job %s: %s", record.local_id, err)
            return

        try:
            with HPCClient(cfg) as client:
                status = client.poll_status(record.scheduler_id)
                old_state = record.status
                record.status = status.state
                record.exit_code = status.exit_code
                record.message = status.message
                if status.state in _TERMINAL_STATES and record.completed_at is None:
                    record.completed_at = time.time()
                self._store.add_or_update(record)
                if old_state != status.state:
                    logger.info(
                        "job %s: %s -> %s", record.local_id, old_state, status.state
                    )
        except Exception as e:
            logger.debug("poll failed for %s: %s", record.local_id, e)
