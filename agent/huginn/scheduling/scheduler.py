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
import contextlib
import logging
import os
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any

from huginn.persistence.campaign import (
    CampaignStoreBackend,
    JobRecord,
    NullCampaignStore,
)
from huginn.scheduling.hpc_queues import (
    HpcQueueLayer,
    ReconcileReport,
    reconcile_layers,
)
from huginn.tools.profile import CostTier

logger = logging.getLogger(__name__)


class ResourceExhausted(Exception):  # noqa: N818
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
    # "cumulative" (default): budget is a session cap — charged on acquire and
    #   never released. Good for a bounded short run.
    # "current": budget expresses concurrent occupancy — released back on
    #   release(). Required for long-running (month/year) campaigns where a
    #   monotonic cumulative counter would hit the cap within hours.
    budget_mode: str = "cumulative"

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
                logger.debug("best-effort op failed", exc_info=True)
                return None

        return cls(
            max_concurrent_heavy=_env_int("HUGINN_MAX_HEAVY_CONCURRENT", 2),
            max_concurrent_light=_env_int("HUGINN_MAX_LIGHT_CONCURRENT", 8),
            cpu_hour_budget=_env_float("HUGINN_CPU_HOUR_BUDGET"),
            gpu_hour_budget=_env_float("HUGINN_GPU_HOUR_BUDGET"),
            budget_mode=os.environ.get("HUGINN_BUDGET_MODE", "cumulative"),
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
        hpc_layer: HpcQueueLayer | None = None,
    ) -> None:
        self.store: CampaignStoreBackend = (
            store if store is not None else NullCampaignStore()
        )
        self.policy: AdmissionPolicy = (
            policy if policy is not None else AdmissionPolicy.from_env()
        )
        # P1: optional HPC queue layer. When set, long-running async jobs
        # additionally arbitrate remote admission before running.
        self.hpc_layer = hpc_layer
        self._heavy_sem = asyncio.Semaphore(self.policy.max_concurrent_heavy)
        self._light_sem = asyncio.Semaphore(self.policy.max_concurrent_light)
        self._budget_lock = threading.Lock()
        self._cpu_hours_used = 0.0
        self._gpu_hours_used = 0.0
        # Records which active jobs have been charged to the budget (for
        # cross-layer reconciliation of the budget layer).
        self._charged_active: set[str] = set()
        # P2: last-progress heartbeat per live job. A job that holds a slot but
        # stops updating this is a "harmonic" resident — occupying resources
        # without making progress (invisible to a plain capacity check).
        self._heartbeats: dict[str, float] = {}
        # async-job path
        self._queue: deque[_QueuedJob] = deque()
        self._live_tasks: dict[str, asyncio.Task] = {}
        # job_id -> 原始 factory 引用. 记录以便 resume_orphaned() 在恢复 orphaned
        # 作业时复用同一工厂, 而不必外部重新 attach.
        self._factories: dict[str, Callable[[], Coroutine[Any, Any, Any]]] = {}
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
            # In "current" mode, still release the charged budget even when the
            # call held no tier semaphore (e.g. a none-tier tool that charged the
            # report budget).
            self._release_budget(admission)
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
        # Free the charged budget only after the tier slot is released, so
        # "current" occupancy accurately reflects in-flight work.
        self._release_budget(admission)

    def _release_budget(self, admission: Admission) -> None:
        """Release charged budget back when policy is in "current" mode."""
        if self.policy.budget_mode != "current":
            return
        with self._budget_lock:
            self._cpu_hours_used = max(
                0.0, self._cpu_hours_used - admission.requested_cpu_hours
            )
            self._gpu_hours_used = max(
                0.0, self._gpu_hours_used - admission.requested_gpu_hours
            )

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
                    "cpu",
                    self._cpu_hours_used,
                    self.policy.cpu_hour_budget,
                    requested_cpu,
                )
            if (
                self.policy.gpu_hour_budget is not None
                and self._gpu_hours_used + requested_gpu > self.policy.gpu_hour_budget
            ):
                raise ResourceExhausted(
                    "gpu",
                    self._gpu_hours_used,
                    self.policy.gpu_hour_budget,
                    requested_gpu,
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
        gpu_hours = None
        if cost is not None:
            cores = cost.get("cpu_hours")
            gpu_hours = cost.get("gpu_hours")
        record = JobRecord(
            job_id=job_id,
            tool_name=tool_name,
            status="queued",
            cost_tier=cost_tier,
            campaign_id=campaign_id,
            working_dir=working_dir,
            compute_action=compute_action,
            cores_requested=cores,
            gpu_hours_requested=gpu_hours,
            queue_position=self.store.next_queue_position(),
        )
        self.store.upsert_job(record)
        self._queue.append(_QueuedJob(job_id, tool_name, cost_tier, cost, factory))
        # 记录 factory 引用, 供 resume_orphaned() 恢复同类 orphaned 作业使用.
        self._factories[job_id] = factory
        self._wake_drainer()
        return job_id

    async def cancel(self, job_id: str) -> bool:
        """真正取消一个 queued / running 作业, 返回是否成功取消.

        优先级:
          - running 作业: 取消其 asyncio Task (由 _run_job 兜底置 status).
          - queued 作业: 直接从内存队列摘除, 不再被 drainer 吸收.
          - 退化情况: 不在内存里但持久层仍是 queued/orphaned → 直接置 cancelled.
        无论哪条路径, 都把 JobRecord.status 持久化为 cancelled.

        未知或已终态的 job_id 返回 False.
        """
        # 1) running: 取消 live task
        task = self._live_tasks.get(job_id)
        if task is not None:
            if not task.done():
                task.cancel()
            self._mark_cancelled(job_id)
            return True

        # 2) queued: 从内存队列摘除
        for i, qj in enumerate(self._queue):
            if qj.job_id == job_id:
                del self._queue[i]
                self._mark_cancelled(job_id)
                return True

        # 3) 退化: job 不在内存但持久层仍待跑 (e.g. recover 后未重挂内存) → 可取消
        record = self.store.get_job(job_id)
        if record is not None and record.status in ("queued", "orphaned"):
            self._mark_cancelled(job_id)
            return True

        return False

    def _mark_cancelled(self, job_id: str) -> None:
        """把持久层 JobRecord 置为 cancelled 并落库."""
        record = self.store.get_job(job_id)
        if record is None:
            return
        record.status = "cancelled"
        record.finished_at = time.time()
        record.error = "cancelled"
        self._factories.pop(job_id, None)
        self.store.upsert_job(record)

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
            logger.debug("best-effort op failed", exc_info=True)
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
        # No running loop — drainer not started; queue still persists to store.
        with contextlib.suppress(RuntimeError):
            self._drainer_wake.set()

    async def _drain(self) -> None:
        """FIFO drainer: admit queued jobs as heavy slots free."""
        while not self._stopped:
            if not self._queue:
                self._drainer_wake.clear()
                try:
                    # Sleep until woken or a slot might have freed.
                    await asyncio.wait_for(self._drainer_wake.wait(), timeout=1.0)
                except TimeoutError:
                    logger.debug("best-effort op failed", exc_info=True)
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
        hpc_held = False
        cancelled = False
        try:
            # P1: acquire a remote admission slot before running when an HPC
            # layer is wired. Blocks (backpressure) while the cluster is full.
            if self.hpc_layer is not None and self.hpc_layer.enabled:
                await self.hpc_layer.acquire_slot(job.job_id)
                hpc_held = True
            # Track this long job against the budget for cross-layer
            # reconciliation (current-mode occupancy).
            if self.policy.budget_mode == "current":
                self._charged_active.add(job.job_id)
            # P2: initial heartbeat — the job is admitted and now expected to
            # make progress (or call touch() periodically).
            self._heartbeats[job.job_id] = time.time()
            result = await job.factory()
        except asyncio.CancelledError:
            # 真取消: 由 cancel() 触发. CancelledError 是 BaseException, 不走下面的
            # Exception 分支, 这里标记后 re-raise, 让 finally 把 status 置为 cancelled.
            cancelled = True
            raise
        except Exception as exc:  # noqa: BLE001 — persist whatever failed
            error = repr(exc)
            logger.exception("queued job %s failed", job.job_id)
        finally:
            if self.policy.budget_mode == "current":
                self._charged_active.discard(job.job_id)
            self._heartbeats.pop(job.job_id, None)
            if hpc_held:
                self.hpc_layer.release_slot(job.job_id)  # type: ignore[union-attr]
            if sem is not None:
                with contextlib.suppress(ValueError):
                    sem.release()
            self._live_tasks.pop(job.job_id, None)
            self._factories.pop(job.job_id, None)
            now = time.time()
            if record is not None:
                if cancelled:
                    record.status = "cancelled"
                else:
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
        - In ``current`` budget mode, orphaned jobs request machine-hours at a
          remote layer that keeps running across the restart; we fold their
          requested hours back into the concurrent-occupancy budget so a fresh
          process doesn't see a zeroed account and flood past the cap.

        Returns a small summary for logging.
        """
        orphaned = 0
        requeued = 0
        for rec in self.store.list_jobs_by_status("running"):
            # These were admitted before the crash; in current mode their
            # machine-hours are still burning remotely, so recount them.
            if self.policy.budget_mode == "current":
                self._count_orphaned(rec)
            rec.status = "orphaned"
            rec.finished_at = time.time()
            rec.error = "orphaned: no live task on restart"
            self.store.upsert_job(rec)
            orphaned += 1
        for _rec in self.store.list_queued_jobs():
            # Re-enqueue without a fresh factory — the caller must re-attach the
            # factory via submit_async resume, or the job_tool re-submits. We
            # keep the record so poll_job still reflects "queued".
            requeued += 1
        return {"orphaned": orphaned, "requeued": requeued}

    def resume_orphaned(
        self,
        factory_provider: (
            Callable[[str, JobRecord], Callable[[], Coroutine[Any, Any, Any]] | None]
            | None
        ) = None,
    ) -> int:
        """一次性 resume 已 orphaned 作业: 重新排队并交给 drainer 继续跑.

        ``recover()`` 会把崩溃遗留的 running 标记为 orphaned; 本方法把它们
        重新插回内存队列。factory 优先级:
            1. 外部传入的 ``factory_provider(job_id, record)`` —— 拿到就用;
            2. 回退到提交时记录的原始 factory 引用 (``self._factories``);
            3. 都没有则该作业跳过(无法恢复), 计数不计入。

        返回成功重新排队的数量。
        """
        resumed = 0
        for rec in self.store.list_jobs_by_status("orphaned"):
            factory = (
                factory_provider(rec.job_id, rec)
                if factory_provider is not None
                else None
            )
            if factory is None:
                factory = self._factories.get(rec.job_id)
            if factory is None:
                # 没有 factory 无法恢复, 保持 orphaned 状态等外部 re-submit.
                logger.warning("no factory to resume orphaned job %s", rec.job_id)
                continue
            rec.status = "queued"
            rec.finished_at = None
            rec.error = None
            if rec.queue_position is None:
                rec.queue_position = self.store.next_queue_position()
            self.store.upsert_job(rec)
            cost = {
                "cpu_hours": rec.cores_requested,
                "gpu_hours": rec.gpu_hours_requested,
            }
            self._queue.append(
                _QueuedJob(rec.job_id, rec.tool_name, rec.cost_tier, cost, factory)
            )
            resumed += 1
        self._wake_drainer()
        return resumed

    def _count_orphaned(self, rec: JobRecord) -> None:
        """Fold an orphaned job's requested hours back into the budget."""
        with self._budget_lock:
            if rec.cores_requested is not None:
                self._cpu_hours_used += rec.cores_requested
            if rec.gpu_hours_requested is not None:
                self._gpu_hours_used += rec.gpu_hours_requested

    def touch(self, job_id: str) -> None:
        """Mark ``job_id`` as making progress (P2 heartbeat).

        Long-running job coroutines should call this periodically; the
        harmonic-component detector (``stalled_jobs``) flags any live job whose
        heartbeat goes stale while it still holds a slot.
        """
        self._heartbeats[job_id] = time.time()

    def stall_timeout(self, job_id: str) -> float | None:
        """Seconds since ``job_id`` last made progress, or None if not tracked."""
        last = self._heartbeats.get(job_id)
        return None if last is None else time.time() - last

    def stalled_jobs(self, timeout: float = 900.0) -> list[str]:
        """Harmonic-component detector: live jobs holding a slot but idle.

        Returns ``job_id`` of tracked jobs whose heartbeat is older than
        ``timeout`` seconds. These are the "invisible" residents — a capacity
        check alone cannot see them, but they tie up a heavy/HPC slot.
        """
        now = time.time()
        return [
            job_id
            for job_id, last in list(self._heartbeats.items())
            if now - last > timeout
        ]

    def reconcile(self) -> ReconcileReport:
        """Cross-layer reconciliation: local vs HPC vs budget state.

        Detects Čech H¹-style consistency obstructions between the scheduler's
        three layers — local job state, the HPC queue layer, and the budget
        occupancy — so a job that is "running" locally but absent on the cluster
        (or charged to budget but not active) is surfaced instead of drifting.
        """
        local: dict[str, str] = {}
        for status in ("queued", "running"):
            for rec in self.store.list_jobs_by_status(status):
                local[rec.job_id] = rec.status
        hpc = self.hpc_layer.snapshot() if self.hpc_layer is not None else {}
        charged = set(self._charged_active)
        return reconcile_layers(local, hpc, charged)

    def resource_contention(self) -> dict[str, list[str]]:
        """Explicit shared-resource contention (P3). Delegates to the HPC layer."""
        if self.hpc_layer is None or not self.hpc_layer.enabled:
            return {}
        return self.hpc_layer.resource_contention()

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
