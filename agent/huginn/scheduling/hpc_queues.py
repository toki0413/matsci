"""HPC queue layer abstraction + cross-layer reconciliation (P1).

Multilayer scheduling on top of the local ``ToolScheduler`` semaphore layer:

- The local layer (heavy/light semaphores) arbitrates *local* concurrency and
  resource budget.
- An ``HpcQueueLayer`` (one per visible cluster) arbitrates *remote* admission
  (queue/partition slots). A long-running job is only fully "running" when it
  holds BOTH a local slot AND an HPC admission slot.

Reconciliation
--------------
Each layer keeps its own view of a job's state. When views disagree — a job
"running" locally but absent/rejected on the cluster, or admitted to the
cluster but untracked locally — the cross-layer view has a topological
consistency violation. In sheaf terms this is a Čech H¹ obstruction: each
layer is locally consistent, but the layers cannot be glued into a single
globally-consistent schedule. ``reconcile_layers`` detects these obstructions
so the caller can act (requeue, cancel, or surface to the operator) instead of
letting each layer drift independently.

HPC status codes are intentionally kept coarse so the comparison is robust to
scheduler-specific state strings (SLURM ``PENDING/RUNNING/...``, PBS ``Q/R/C``…).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)

# Coarse per-layer job states used by reconciliation. Layers map their native
# states onto these.
LOCAL_ACTIVE = {"running", "queued"}
HPC_ACTIVE = {"queued", "running"}


@dataclass
class HpcQueueConfig:
    """Per-cluster admission + submission config for one HPC queue layer.

    ``max_concurrent`` caps how many jobs this scheduler admits to the cluster
    at once — the remote analogue of ``AdmissionPolicy.max_concurrent_heavy``.
    """

    name: str  # cluster / queue logical name
    max_concurrent: int = 2
    queue: str | None = None  # partition (SLURM) / queue (PBS) override
    walltime: str | None = None
    nodes: int | None = None
    ntasks_per_node: int | None = None
    gpus_per_node: int = 0
    modules: list[str] = field(default_factory=list)
    env_vars: dict[str, str] = field(default_factory=dict)


class HpcQueueLayer:
    """Admission layer for one HPC cluster.

    ``client`` is any object exposing ``submit_job(script, job_name) -> remote_id``
    and ``poll_status(remote_id) -> JobStatus`` (e.g. ``huginn.hpc.client.HPCClient``).
    When ``client`` is ``None`` the layer is disabled and returns ``False`` from
    ``acquire_slot`` — the scheduler degrades to local-only behaviour (backward
    compatible).

    Slot state is tracked in memory keyed by the *local* job id. It is not
    persisted in ``JobRecord`` (avoids schema migration); recovery of remote
    state is the HPC monitor's job, this layer only arbitrates admission.
    """

    def __init__(self, config: HpcQueueConfig, client: Any | None = None) -> None:
        self.config = config
        self.client = client
        self._sem: asyncio.Semaphore | None = None
        # local_job_id -> coarse state: "reserved" | "submitted" | "running"
        self._slots: dict[str, str] = {}
        # local_job_id -> remote cluster job id
        self._remote: dict[str, str] = {}
        # P3: local_job_id -> set of shared resource names (queue, scratch path,
        #   license server, …). Two jobs sharing a resource are a hyperedge on
        #   that resource — an explicit representation of contention.
        self._resources: dict[str, set[str]] = {}

    # ── enablement ────────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        """True when a client is wired and remote admission is meaningful."""
        return self.client is not None

    # ── admission ─────────────────────────────────────────────────────────

    async def acquire_slot(
        self, job_id: str, shared_resources: list[str] | None = None
    ) -> bool:
        """Reserve a remote admission slot for ``job_id``.

        Returns ``False`` when the layer is disabled (local-only mode). Blocks
        (natural backpressure) until a slot frees when the cluster is saturated.
        ``shared_resources`` records which shared resources the job contends for
        (see ``resource_contention``).
        """
        if not self.enabled:
            return False
        if self._sem is None:
            self._sem = asyncio.Semaphore(self.config.max_concurrent)
        await self._sem.acquire()
        self._slots[job_id] = "reserved"
        if shared_resources:
            self._resources[job_id] = set(shared_resources)
        return True

    def release_slot(self, job_id: str) -> None:
        """Free the remote slot held by ``job_id``. Safe to call once."""
        if not self.enabled or self._sem is None:
            return
        self._slots.pop(job_id, None)
        self._remote.pop(job_id, None)
        self._resources.pop(job_id, None)
        try:
            self._sem.release()
        except ValueError:
            # over-release guard — mirrors ToolScheduler.release
            logger.warning("HPC slot over-release for %s", job_id)

    # ── lifecycle transitions ─────────────────────────────────────────────

    def mark_submitted(self, job_id: str, remote_id: str) -> None:
        """Record that ``job_id`` was submitted to the cluster as ``remote_id``."""
        self._slots[job_id] = "submitted"
        self._remote[job_id] = remote_id

    def mark_running(self, job_id: str) -> None:
        """Upgrade a submitted job to running (client poll reported RUNNING)."""
        if job_id in self._slots:
            self._slots[job_id] = "running"

    # ── observability ─────────────────────────────────────────────────────

    def state_of(self, job_id: str) -> str | None:
        """Coarse HPC state of ``job_id``, or None if not tracked."""
        return self._slots.get(job_id)

    def remote_id(self, job_id: str) -> str | None:
        return self._remote.get(job_id)

    def snapshot(self) -> dict[str, str]:
        """Snapshot of all tracked local-job → HPC-state mappings."""
        return dict(self._slots)

    def resource_contention(self) -> dict[str, list[str]]:
        """Explicit shared-resource contention (P3 hyperedges).

        Returns ``resource -> [job_id, …]`` for every shared resource that more
        than one tracked job currently contends for. This surfaces the
        higher-order competition that a flat capacity check cannot see — e.g.
        two jobs sharing the same scratch disk or license server.
        """
        buckets: dict[str, list[str]] = {}
        for job_id, resources in self._resources.items():
            for res in resources:
                buckets.setdefault(res, []).append(job_id)
        return {res: jobs for res, jobs in buckets.items() if len(jobs) > 1}

    def submit(self, job_id: str, script: str, job_name: str) -> str:
        """Submit ``script`` to the cluster. Returns the remote job id.

        Raises ``RuntimeError`` if the layer has no client.
        """
        if not self.enabled:
            raise RuntimeError("HpcQueueLayer is disabled (no client)")
        remote_id = self.client.submit_job(script, job_name=job_name)
        self.mark_submitted(job_id, remote_id)
        return remote_id


# ── reconciliation ────────────────────────────────────────────────────────


@dataclass
class ReconcileIssue:
    """One cross-layer consistency violation (a Čech H¹ obstruction)."""

    kind: Literal[
        "local_active_hpc_absent",
        "hpc_active_local_absent",
        "budget_charged_not_active",
    ]
    job_id: str
    severity: Literal["critical", "warning"]
    detail: str


@dataclass
class ReconcileReport:
    """Aggregate result of a cross-layer reconciliation pass."""

    issues: list[ReconcileIssue] = field(default_factory=list)

    @property
    def consistent(self) -> bool:
        return not self.issues

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for issue in self.issues:
            counts[issue.kind] = counts.get(issue.kind, 0) + 1
        return counts


def reconcile_layers(
    local_states: dict[str, str],
    hpc_states: dict[str, str],
    budget_charged: set[str] | None = None,
) -> ReconcileReport:
    """Detect cross-layer consistency violations between local and HPC views.

    Args:
        local_states: map ``job_id -> local status`` (e.g. "queued"/"running"/
            "finished"/"failed"/"orphaned").
        hpc_states: map ``job_id -> coarse HPC state`` (from ``HpcQueueLayer.snapshot``).
        budget_charged: set of ``job_id`` whose cost was charged to the budget but
            that are not actually active (local queued/running). Defaults to None
            (skip the budget check).

    Detects three obstruction classes:
      - ``local_active_hpc_absent``: local queued/running but the HPC layer has
        no record — the job never reached (or dropped off) the cluster.
      - ``hpc_active_local_absent``: HPC layer tracks it as active but local is
        not active (job finished/failed locally or was never tracked) — the
        cluster runs a job the scheduler no longer accounts for.
      - ``budget_charged_not_active``: budget was charged but the job is not
        active locally — charged compute that is not running.
    """
    report = ReconcileReport()

    for job_id, status in local_states.items():
        if status in LOCAL_ACTIVE:
            hpc_state = hpc_states.get(job_id)
            if hpc_state is None:
                report.issues.append(
                    ReconcileIssue(
                        kind="local_active_hpc_absent",
                        job_id=job_id,
                        severity="critical",
                        detail=(
                            f"local '{status}' but HPC layer has no record of job "
                            f"(never submitted or dropped off cluster)"
                        ),
                    )
                )
            elif hpc_state not in HPC_ACTIVE:
                report.issues.append(
                    ReconcileIssue(
                        kind="local_active_hpc_absent",
                        job_id=job_id,
                        severity="warning",
                        detail=f"local '{status}' but HPC state is '{hpc_state}'",
                    )
                )

    for job_id, hpc_state in hpc_states.items():
        if hpc_state in HPC_ACTIVE and local_states.get(job_id) not in LOCAL_ACTIVE:
            report.issues.append(
                ReconcileIssue(
                    kind="hpc_active_local_absent",
                    job_id=job_id,
                    severity="critical",
                    detail=(
                        f"HPC state '{hpc_state}' but job is not active locally "
                        f"(local state: {local_states.get(job_id, '<untracked>')}) — "
                        f"cluster may be running a job the scheduler forgot about"
                    ),
                )
            )

    if budget_charged is not None:
        for job_id in budget_charged:
            if local_states.get(job_id) not in LOCAL_ACTIVE:
                report.issues.append(
                    ReconcileIssue(
                        kind="budget_charged_not_active",
                        job_id=job_id,
                        severity="warning",
                        detail=(
                            "budget charged but job not active locally "
                            f"(local state: {local_states.get(job_id, '<untracked>')})"
                        ),
                    )
                )

    return report
