"""Tests for the P1 HPC queue layer + cross-layer reconciliation.

Covers:
- ``reconcile_layers`` obstruction detection (local-active w/o HPC, HPC-active
  w/o local, budget-charged-not-active).
- ``HpcQueueLayer`` admission cap + disable (local-only) behaviour.
- ``ToolScheduler`` integration: injecting an HPC layer makes async jobs
  arbitrate a remote slot; ``reconcile()`` surfaces obstructions.
- P0 budget "current" mode: released back on release().
- P2 harmonic-component detector (``stalled_jobs`` / ``touch``).
- P3 explicit shared-resource contention (``resource_contention``).
- ``AgentFactory._build_hpc_layer`` wiring (disabled when local-only).

Follows the same asyncio.run pattern as ``test_tool_scheduler.py``.
"""

from __future__ import annotations

import asyncio

import pytest

from huginn.persistence.campaign import NullCampaignStore, SqliteCampaignStore
from huginn.scheduling import (
    AdmissionPolicy,
    HpcQueueConfig,
    HpcQueueLayer,
    ToolScheduler,
    reconcile_layers,
)

# ── reconcile_layers: pure obstruction detection ───────────────────────────


def test_reconcile_consistent_when_views_agree():
    report = reconcile_layers(
        local_states={"j1": "running", "j2": "finished"},
        hpc_states={"j1": "running"},
    )
    assert report.consistent
    assert report.issues == []


def test_reconcile_detects_local_active_hpc_absent():
    report = reconcile_layers(
        local_states={"j1": "running"},
        hpc_states={},
    )
    assert not report.consistent
    kinds = [i.kind for i in report.issues]
    assert "local_active_hpc_absent" in kinds
    assert report.issues[0].severity == "critical"


def test_reconcile_detects_hpc_active_local_absent():
    report = reconcile_layers(
        local_states={"j1": "finished"},
        hpc_states={"j1": "running"},
    )
    kinds = [i.kind for i in report.issues]
    assert "hpc_active_local_absent" in kinds


def test_reconcile_detects_budget_charged_not_active():
    report = reconcile_layers(
        local_states={"j1": "finished"},
        hpc_states={},
        budget_charged={"j1"},
    )
    kinds = [i.kind for i in report.issues]
    assert "budget_charged_not_active" in kinds


def test_reconcile_ignores_finished_local():
    # A finished job is not active anywhere → no obstruction.
    report = reconcile_layers(
        local_states={"j1": "finished"},
        hpc_states={},
    )
    assert report.consistent


# ── HpcQueueLayer: admission + disable ─────────────────────────────────────


class _FakeClient:
    """Minimal stand-in for HPCClient.submit_job / poll_status."""

    def __init__(self) -> None:
        self.submitted: list[str] = []

    def submit_job(self, script: str, job_name: str = "job") -> str:
        self.submitted.append(job_name)
        return f"remote_{len(self.submitted)}"

    def poll_status(self, remote_id: str):
        return None


def test_hpc_layer_disabled_without_client():
    layer = HpcQueueLayer(HpcQueueConfig(name="cluster", max_concurrent=2), client=None)
    assert layer.enabled is False

    async def run() -> None:
        assert await layer.acquire_slot("j1") is False

    asyncio.run(run())


def test_hpc_layer_enforces_concurrency_cap():
    client = _FakeClient()
    layer = HpcQueueLayer(HpcQueueConfig(name="cluster", max_concurrent=2), client=client)
    assert layer.enabled is True

    async def run() -> None:
        assert await layer.acquire_slot("j1")
        assert await layer.acquire_slot("j2")
        # 3rd slot must block.
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(layer.acquire_slot("j3"), timeout=0.1)
        layer.release_slot("j1")
        # Now j3 can acquire.
        assert await asyncio.wait_for(layer.acquire_slot("j3"), timeout=0.5)

    asyncio.run(run())


def test_hpc_layer_submit_and_state():
    client = _FakeClient()
    layer = HpcQueueLayer(HpcQueueConfig(name="cluster"), client=client)
    remote = layer.submit("job1", "#!/bin/bash\necho hi", job_name="relax")
    assert remote == "remote_1"
    assert layer.state_of("job1") == "submitted"
    layer.mark_running("job1")
    assert layer.state_of("job1") == "running"
    assert layer.snapshot() == {"job1": "running"}


# ── ToolScheduler + HpcQueueLayer integration ──────────────────────────────


def test_scheduler_async_runs_through_hpc_slot():
    client = _FakeClient()
    layer = HpcQueueLayer(HpcQueueConfig(name="cluster"), client=client)
    scheduler = ToolScheduler(
        store=NullCampaignStore(),
        policy=AdmissionPolicy(max_concurrent_heavy=2, max_concurrent_light=8),
        hpc_layer=layer,
    )

    async def run() -> None:
        scheduler.start()
        try:
            done = asyncio.Event()

            async def job() -> str:
                done.set()
                return "ok"

            jid = await scheduler.submit_async("vasp_tool", "heavy", None, job)
            await asyncio.wait_for(done.wait(), timeout=1.0)
            await asyncio.sleep(0.1)  # let _run_job finish + release slot
            # The job did acquire + release the HPC slot cleanly.
            assert layer.snapshot() == {}
        finally:
            scheduler.stop()

    asyncio.run(run())


def test_scheduler_reconcile_flags_drift():
    client = _FakeClient()
    layer = HpcQueueLayer(HpcQueueConfig(name="cluster"), client=client)
    scheduler = ToolScheduler(
        store=NullCampaignStore(),
        policy=AdmissionPolicy(max_concurrent_heavy=2, max_concurrent_light=8),
        hpc_layer=layer,
    )
    # Simulate a job the local store thinks is running but the HPC layer never
    # saw (dropped before submit).
    import time

    from huginn.persistence.campaign import JobRecord

    scheduler.store.upsert_job(
        JobRecord(
            job_id="j_drift", tool_name="vasp_tool", status="running",
            cost_tier="heavy", started_at=time.time(),
        )
    )
    report = scheduler.reconcile()
    kinds = [i.kind for i in report.issues]
    assert "local_active_hpc_absent" in kinds


# ── P0: budget "current" mode releases back ────────────────────────────────


def test_budget_current_mode_releases_on_release(tmp_path):
    store = SqliteCampaignStore(tmp_path / "c.sqlite")
    scheduler = ToolScheduler(
        store=store,
        policy=AdmissionPolicy(
            max_concurrent_heavy=5,
            max_concurrent_light=8,
            cpu_hour_budget=10.0,
            budget_mode="current",
        ),
    )

    async def run() -> None:
        a1 = await scheduler.acquire("vasp_tool", "heavy", {"cpu_hours": 6.0})
        assert scheduler.snapshot().cpu_hours_used == 6.0
        # 6h + 6h = 12h > 10h still rejects while held.
        from huginn.scheduling import ResourceExhausted

        with pytest.raises(ResourceExhausted):
            await scheduler.acquire("vasp_tool", "heavy", {"cpu_hours": 6.0})
        # Release frees the budget back → the same 6h becomes affordable again.
        scheduler.release(a1)
        assert scheduler.snapshot().cpu_hours_used == 0.0
        a2 = await scheduler.acquire("vasp_tool", "heavy", {"cpu_hours": 6.0})
        scheduler.release(a2)

    asyncio.run(run())
    store.close()


def test_budget_cumulative_mode_does_not_release():
    scheduler = ToolScheduler(
        store=NullCampaignStore(),
        policy=AdmissionPolicy(
            max_concurrent_heavy=5,
            max_concurrent_light=8,
            cpu_hour_budget=10.0,  # default budget_mode="cumulative"
        ),
    )

    async def run() -> None:
        a1 = await scheduler.acquire("vasp_tool", "heavy", {"cpu_hours": 6.0})
        scheduler.release(a1)
        # Cumulative: budget stays charged even after release.
        assert scheduler.snapshot().cpu_hours_used == 6.0

    asyncio.run(run())


# ── P3: explicit shared-resource contention ────────────────────────────────


def test_resource_contention_detection():
    client = _FakeClient()
    # 3 concurrent jobs → need ≥3 slots so none blocks on the semaphore.
    layer = HpcQueueLayer(
        HpcQueueConfig(name="cluster", max_concurrent=3), client=client
    )

    async def run() -> None:
        await layer.acquire_slot("j1", shared_resources=["/scratch/a", "feat_license"])
        await layer.acquire_slot("j2", shared_resources=["/scratch/a"])
        await layer.acquire_slot("j3", shared_resources=["/scratch/b"])

        contention = layer.resource_contention()
        # /scratch/a is shared by j1+j2; /scratch/b and the license are not.
        assert contention == {"/scratch/a": ["j1", "j2"]}

        # Releasing j1 removes its claim → no more contention on /scratch/a.
        layer.release_slot("j1")
        assert layer.resource_contention() == {}

    asyncio.run(run())


def test_resource_contention_empty_when_no_layer():
    scheduler = ToolScheduler(
        store=NullCampaignStore(),
        policy=AdmissionPolicy(max_concurrent_heavy=2),
    )
    assert scheduler.resource_contention() == {}


# ── P2: harmonic-component detector (stalled jobs) ─────────────────────────


def test_stalled_jobs_detects_idle_resident():
    scheduler = ToolScheduler(
        store=NullCampaignStore(),
        policy=AdmissionPolicy(max_concurrent_heavy=2),
    )
    # Simulate a live job admitted with an initial heartbeat.
    scheduler._heartbeats["j_busy"] = 0.0  # noqa: SLF001 — direct state for test
    scheduler.touch("j_busy")  # refreshed → not stalled
    scheduler._heartbeats["j_stuck"] = 0.0  # noqa: SLF001 — never refreshed

    stalled = scheduler.stalled_jobs(timeout=1.0)
    assert "j_stuck" in stalled
    assert "j_busy" not in stalled


def test_async_job_cleans_up_heartbeat():
    layer = HpcQueueLayer(HpcQueueConfig(name="cluster"), client=_FakeClient())
    scheduler = ToolScheduler(
        store=NullCampaignStore(),
        policy=AdmissionPolicy(max_concurrent_heavy=2),
        hpc_layer=layer,
    )

    async def run() -> None:
        scheduler.start()
        try:
            done = asyncio.Event()

            async def job() -> str:
                done.set()
                return "ok"

            jid = await scheduler.submit_async("vasp_tool", "heavy", None, job)
            await asyncio.wait_for(done.wait(), timeout=1.0)
            await asyncio.sleep(0.1)
            # After finish, heartbeat is removed and no slot is held.
            assert jid not in scheduler._heartbeats  # noqa: SLF001
            assert not scheduler.stalled_jobs(timeout=0.0)
        finally:
            scheduler.stop()

    asyncio.run(run())


# ── AgentFactory HPC wiring ────────────────────────────────────────────────


def test_factory_build_hpc_layer_disabled_when_local():
    from huginn.agents.factory import AgentFactory
    from huginn.config import HuginnConfig

    cfg = HuginnConfig(provider="ollama", model="qwen2.5:14b", workspace="/tmp/ws")
    # Default hpc_scheduler="local" → no HPC layer.
    obj = object.__new__(AgentFactory)
    obj.config = cfg
    assert obj._build_hpc_layer() is None


def test_factory_build_hpc_layer_enabled_when_remote():
    from huginn.agents.factory import AgentFactory
    from huginn.config import HuginnConfig

    cfg = HuginnConfig(
        provider="ollama",
        model="qwen2.5:14b",
        workspace="/tmp/ws",
        hpc_scheduler="slurm",
        hpc_host="login.cluster",
        hpc_username="researcher",
    )
    obj = object.__new__(AgentFactory)
    obj.config = cfg
    layer = obj._build_hpc_layer()
    assert layer is not None
    assert layer.enabled is True
    assert layer.config.name == "login.cluster"
