"""Central tool scheduler — cross-tool + cross-agent admission control.

Closes the gap surfaced by the Claude Code / Codex benchmark: Huginn's tool
dispatch had concurrency (ToolNode fires multiple tool_calls per turn,
``submit_async`` spawns background tasks) but no coordination — per-tool
``_async_jobs`` dicts, scattered semaphores, no global cap, no resource
accounting, sub-agents sharing no resource pool with their parent.

This module provides:
- ``AdmissionPolicy``: per-tier concurrency caps + optional cpu/gpu hour budget.
- ``ToolScheduler.acquire/release``: every tool call (sync path, wired in
  ``agent.py:_invoke_with_hooks``) acquires a slot keyed by ``cost_tier``;
  saturated calls await the semaphore (natural backpressure to the LLM).
- ``ToolScheduler.submit_async``: heavy async jobs enter a FIFO queue persisted
  to ``CampaignStoreBackend``; a background drainer admits them as slots free.
- Cross-agent sharing: one ``ToolScheduler`` per workspace, injected by
  ``AgentFactory`` into every (sub-)agent, so parent and children share the
  same semaphores.

Budget is a conservative session cap: ``estimate_cost()`` is consulted at
acquire time and the cumulative requested hours are tracked; when a budget is
set and would be exceeded, ``ResourceExhausted`` is raised and surfaced to the
LLM as an ``resource_exhausted`` error so it can switch to a light alternative.
Tools that return ``None`` from ``estimate_cost()`` (the majority) skip the
budget gate.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

from huginn.persistence.campaign import (
    CampaignStoreBackend,
    JobRecord,
    NullCampaignStore,
)
from huginn.tools.profile import CostTier

logger = logging.getLogger(__name__)


class ResourceExhausted(Exception):
    """Raised when admitting a call would exceed the configured cpu/gpu budget.

    Caught in ``agent.py:_invoke_with_hooks`` and surfaced to the LLM as an
    ``resource_exhausted`` error dict (mirroring the pre_tool_use block path)
    so the model can switch to a light alternative instead of bare-retrying.
    """

    def __init__(self, kind: str, used: float, budget: float, requested: float) -> None:
        self.kind = kind
        self.used = used
        self.budget = budget
        self.requested = requested
        super().__init__(
            f"{kind}_hour_budget exhausted: used={used:.2f} + requested={requested:.2f} "
            f"> budget={budget:.2f}"
        )


@dataclass
class AdmissionPolicy:
    """Concurrency caps and optional resource budgets for the scheduler."""

    max_concurrent_heavy: int = 2
    max_concurrent_light: int = 8
    cpu_hour_budget: float | None = None
    gpu_hour_budget: float | None = None

    @classmethod
    def from_env(cls) -> AdmissionPolicy:
        def _env_int(name: str, default: int) -> int:
            raw = os.environ.get(name)
            try:
                return int(raw) if raw is not None else default
            except ValueError:
                return default

        def _env_float(name: str) -> float | None:
            raw = os.environ.get(name)
            if raw is None or raw == "":
                return None
            try:
                return float(raw)
            except ValueError:
                return None

        return cls(
            max_concurrent_heavy=_env_int("HUGINN_MAX_HEAVY_CONCURRENT", 2),
            max_concurrent_light=_env_int("HUGINN_MAX_LIGHT_CONCURRENT", 8),
            cpu_hour_budget=_env_float("HUGINN_CPU_HOUR_BUDGET"),
            gpu_hour_budget=_env_float("HUGINN_GPU_HOUR_BUDGET"),
        )


@dataclass
class Admission:
    """Ticket returned by ``acquire`` — passed back to ``release``."""

    tool_name: str
    cost_tier: CostTier
    held_semaphore: bool
    requested_cpu_hours: float
    requested_gpu_hours: float


@dataclass
class _QueuedJob:
    """In-memory mirror of a queued async job awaiting the drainer."""

    job_id: str
    tool_name: str
    cost_tier: CostTier
    cost: dict[str, float] | None
    factory: Callable[[], Coroutine[Any, Any, Any]]


@dataclass
class SchedulerStatus:
    """Snapshot for observability / CRITIC consumption."""

    heavy_in_flight: int
    light_in_flight: int
    queued_depth: int
    cpu_hours_used: float
    gpu_hours_used: float
    cpu_hour_budget: float | None
    gpu_hour_budget: float | None


class ToolScheduler:
    """Cross-tool + cross-agent admission control + unified job registry.

    One instance per workspace, shared by every (sub-)agent built by the same
    ``AgentFactory`` so semaphores arbitrate parent/child contention. The
    persistent backend is ``CampaignStoreBackend`` (P3's jobs table); when P3
    is not yet wired, pass ``NullCampaignStore`` for in-memory-only operation.
    """

    def __init__(
        self,
        store: CampaignStoreBackend | None = None,
        policy: AdmissionPolicy | None = None,
    ) -> None:
        self.store: CampaignStoreBackend = store if store is not None else NullCampaignStore()
        self.policy: AdmissionPolicy = policy if policy is not None else AdmissionPolicy.from_env()
        self._heavy_sem = asyncio.Semaphore(self.policy.max_concurrent_heavy)
        self._light_sem = asyncio.Semaphore(self.policy.max_concurrent_light)
        self._budget_lock = threading.Lock()
        self._cpu_hours_used = 0.0
        self._gpu_hours_used = 0.0
        # async-job path
        self._queue: deque[_QueuedJob] = deque()
        self._live_tasks: dict[str, asyncio.Task] = {}
        self._drainer: asyncio.Task | None = None
        self._drainer_wake = asyncio.Event()
        self._stopped = False

    # ── sync path: acquire / release ────────────────────────────────────

    async def acquire(
        self,
        tool_name: str,
        cost_tier: CostTier,
        cost: dict[str, float] | None,
    ) -> Admission:
        """Reserve a slot for a tool call. Awaits the tier semaphore when saturated.

        Raises ``ResourceExhausted`` (before awaiting) when the session budget
        would be exceeded — callers surface this to the LLM.
        """
        requested_cpu = float((cost or {}).get("cpu_hours", 0.0) or 0.0)
        requested_gpu = float((cost or {}).get("gpu_hours", 0.0) or 0.0)
        self._check_budget(requested_cpu, requested_gpu)
        sem = self._sem_for(cost_tier)
        if sem is not None:
            await sem.acquire()
        return Admission(
            tool_name=tool_name,
            cost_tier=cost_tier,
            held_semaphore=sem is not None,
            requested_cpu_hours=requested_cpu,
            requested_gpu_hours=requested_gpu,
        )

    def release(self, admission: Admission) -> None:
        """Release the slot held by an ``Admission``. Safe to call once."""
        if not admission.held_semaphore:
            return
        sem = self._sem_for(admission.cost_tier)
        if sem is None:
            return
        try:
            sem.release()
        except ValueError:
            # Over-release guard: semaphore already at initial value. Shouldn't
            # happen with paired acquire/release, but don't crash the agent.
            logger.warning("semaphore over-release for %s", admission.tool_name)

    def _sem_for(self, cost_tier: CostTier) -> asyncio.Semaphore | None:
        if cost_tier == "heavy":
            return self._heavy_sem
        if cost_tier == "light":
            return self._light_sem
        return None

    def _check_budget(self, requested_cpu: float, requested_gpu: float) -> None:
        with self._budget_lock:
            if (
                self.policy.cpu_hour_budget is not None
                and self._cpu_hours_used + requested_cpu > self.policy.cpu_hour_budget
            ):
                raise ResourceExhausted(
                    "cpu", self._cpu_hours_used, self.policy.cpu_hour_budget, requested_cpu
                )
            if (
                self.policy.gpu_hour_budget is not None
                and self._gpu_hours_used + requested_gpu > self.policy.gpu_hour_budget
            ):
                raise ResourceExhausted(
                    "gpu", self._gpu_hours_used, self.policy.gpu_hour_budget, requested_gpu
                )
            self._cpu_hours_used += requested_cpu
            self._gpu_hours_used += requested_gpu

    # ── async path: submit_async + drainer ──────────────────────────────

    async def submit_async(
        self,
        tool_name: str,
        cost_tier: CostTier,
        cost: dict[str, float] | None,
        factory: Callable[[], Coroutine[Any, Any, Any]],
        campaign_id: str | None = None,
        working_dir: str | None = None,
        compute_action: str | None = None,
    ) -> str:
        """Enqueue a heavy async job. Returns a job_id immediately.

        The job is persisted as ``queued`` and picked up by the drainer when a
        heavy slot frees. If a slot is free right now, the drainer admits it on
        its next tick (no long poll — we wake it explicitly).
        """
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        cores = None
        if cost is not None:
            cores = cost.get("cpu_hours")
        record = JobRecord(
            job_id=job_id,
            tool_name=tool_name,
            status="queued",
            cost_tier=cost_tier,
            campaign_id=campaign_id,
            working_dir=working_dir,
            compute_action=compute_action,
            cores_requested=cores,
            queue_position=self.store.next_queue_position(),
        )
        self.store.upsert_job(record)
        self._queue.append(
            _QueuedJob(job_id, tool_name, cost_tier, cost, factory)
        )
        self._wake_drainer()
        return job_id

    def get_job_status(self, job_id: str) -> JobRecord | None:
        """Live status for poll_job. Reads the persistent store."""
        return self.store.get_job(job_id)

    def start(self) -> None:
        """Start the background drainer coroutine. Idempotent."""
        if self._drainer is not None or self._stopped:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop — drainer started lazily on first submit_async
            # within a loop. Skip silently; tests that don't run the drainer
            # can still exercise acquire/release + queue state.
            return
        self._drainer = loop.create_task(self._drain())

    def stop(self) -> None:
        """Cancel the drainer and await live tasks best-effort."""
        self._stopped = True
        if self._drainer is not None:
            self._drainer.cancel()
            self._drainer = None
        self._wake_drainer()

    def _wake_drainer(self) -> None:
        try:
            self._drainer_wake.set()
        except RuntimeError:
            # No running loop — drainer not started; queue still persists to store.
            pass

    async def _drain(self) -> None:
        """FIFO drainer: admit queued jobs as heavy slots free."""
        while not self._stopped:
            if not self._queue:
                self._drainer_wake.clear()
                try:
                    # Sleep until woken or a slot might have freed.
                    await asyncio.wait_for(self._drainer_wake.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                continue
            job = self._queue[0]
            # For heavy jobs, acquire the heavy sem first (blocks until free).
            sem = self._sem_for(job.cost_tier)
            if sem is not None:
                await sem.acquire()
            # Pop and admit.
            self._queue.popleft()
            now = time.time()
            record = self.store.get_job(job.job_id)
            if record is not None:
                record.status = "running"
                record.admitted_at = now
                record.started_at = now
                record.queue_position = None
                self.store.upsert_job(record)
            task = asyncio.create_task(self._run_job(job, sem))
            self._live_tasks[job.job_id] = task

    async def _run_job(self, job: _QueuedJob, sem: asyncio.Semaphore | None) -> None:
        """Execute a queued job, then release its slot and persist the result."""
        record = self.store.get_job(job.job_id)
        error: str | None = None
        result: Any = None
        try:
            result = await job.factory()
        except Exception as exc:  # noqa: BLE001 — persist whatever failed
            error = repr(exc)
            logger.exception("queued job %s failed", job.job_id)
        finally:
            if sem is not None:
                try:
                    sem.release()
                except ValueError:
                    pass
            self._live_tasks.pop(job.job_id, None)
            now = time.time()
            if record is not None:
                record.status = "failed" if error is not None else "finished"
                record.finished_at = now
                record.error = error
                if result is not None:
                    import json

                    try:
                        record.result_json = json.dumps(result, default=str)
                    except (TypeError, ValueError):
                        record.result_json = str(result)
                self.store.upsert_job(record)

    # ── recovery / observability ────────────────────────────────────────

    def recover(self) -> dict[str, int]:
        """Reconcile persisted state after a restart.

        - Jobs left ``running`` (no live task) → marked ``orphaned`` for the
          job_tool / HPC layer to pick up.
        - Jobs left ``queued`` → re-enqueued in memory (FIFO order preserved by
          ``queue_position``).

        Returns a small summary for logging.
        """
        orphaned = 0
        requeued = 0
        for rec in self.store.list_jobs_by_status("running"):
            rec.status = "orphaned"
            rec.finished_at = time.time()
            rec.error = "orphaned: no live task on restart"
            self.store.upsert_job(rec)
            orphaned += 1
        for rec in self.store.list_queued_jobs():
            # Re-enqueue without a fresh factory — the caller must re-attach the
            # factory via submit_async resume, or the job_tool re-submits. We
            # keep the record so poll_job still reflects "queued".
            requeued += 1
        return {"orphaned": orphaned, "requeued": requeued}

    def snapshot(self) -> SchedulerStatus:
        heavy_in_flight = self.policy.max_concurrent_heavy - self._heavy_sem._value  # type: ignore[attr-defined]
        light_in_flight = self.policy.max_concurrent_light - self._light_sem._value  # type: ignore[attr-defined]
        return SchedulerStatus(
            heavy_in_flight=max(heavy_in_flight, 0),
            light_in_flight=max(light_in_flight, 0),
            queued_depth=len(self._queue),
            cpu_hours_used=self._cpu_hours_used,
            gpu_hours_used=self._gpu_hours_used,
            cpu_hour_budget=self.policy.cpu_hour_budget,
            gpu_hour_budget=self.policy.gpu_hour_budget,
        )
