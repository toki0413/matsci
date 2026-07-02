"""Regression net for the central ToolScheduler (Tier 1 P6).

Locks the admission/queueing/recovery contract that ``agent.py:_invoke_with_hooks``
and ``AgentFactory._shared_scheduler`` rely on. Follows the same asyncio.run
pattern as ``test_autoloop_engine.py`` — no pytest-asyncio dependency.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from huginn.persistence.campaign import (
    NullCampaignStore,
    SqliteCampaignStore,
)
from huginn.scheduling import (
    Admission,
    AdmissionPolicy,
    ResourceExhausted,
    ToolScheduler,
)


# ── Test 1: admission tiering ──────────────────────────────────────────────


def test_heavy_saturates_light_does_not():
    scheduler = ToolScheduler(
        store=NullCampaignStore(),
        policy=AdmissionPolicy(max_concurrent_heavy=2, max_concurrent_light=8),
    )

    async def run() -> None:
        a1 = await scheduler.acquire("vasp_tool", "heavy", None)
        a2 = await scheduler.acquire("vasp_tool", "heavy", None)
        # 3rd heavy should block; we probe with a short timeout instead of
        # hanging the suite.
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                scheduler.acquire("vasp_tool", "heavy", None), timeout=0.1
            )
        # light tier is a separate semaphore — must NOT be blocked by heavy.
        la = await asyncio.wait_for(
            scheduler.acquire("materials_database_tool", "light", None), timeout=0.5
        )
        scheduler.release(la)
        scheduler.release(a1)
        scheduler.release(a2)

    asyncio.run(run())


# ── Test 2: release wakes a blocked acquire ────────────────────────────────


def test_release_wakes_blocked_acquire():
    scheduler = ToolScheduler(
        store=NullCampaignStore(),
        policy=AdmissionPolicy(max_concurrent_heavy=1, max_concurrent_light=8),
    )

    async def run() -> None:
        a1 = await scheduler.acquire("vasp_tool", "heavy", None)  # holds the only slot
        completed = False

        async def wait_for_slot() -> None:
            nonlocal completed
            a2 = await scheduler.acquire("vasp_tool", "heavy", None)
            completed = True
            scheduler.release(a2)

        task = asyncio.create_task(wait_for_slot())
        await asyncio.sleep(0.05)  # let it park on the semaphore
        assert not completed
        scheduler.release(a1)
        await asyncio.wait_for(task, timeout=1.0)
        assert completed

    asyncio.run(run())


# ── Test 3: resource budget exhaustion ─────────────────────────────────────


def test_cpu_budget_exhaustion_raises():
    scheduler = ToolScheduler(
        store=NullCampaignStore(),
        policy=AdmissionPolicy(
            max_concurrent_heavy=5,
            max_concurrent_light=8,
            cpu_hour_budget=10.0,
        ),
    )

    async def run() -> None:
        # 6h + 6h = 12h > 10h budget → second must reject.
        a1 = await scheduler.acquire("vasp_tool", "heavy", {"cpu_hours": 6.0})
        with pytest.raises(ResourceExhausted) as exc:
            await scheduler.acquire("vasp_tool", "heavy", {"cpu_hours": 6.0})
        assert exc.value.kind == "cpu"
        assert exc.value.budget == 10.0
        scheduler.release(a1)

    asyncio.run(run())


# ── Test 4: cross-agent sharing — one scheduler, two "agents" ──────────────


def test_shared_scheduler_arbitrates_across_agents():
    # The contract AgentFactory relies on: parent and child agents hold the
    # SAME scheduler instance, so their acquire/release share semaphores.
    shared = ToolScheduler(
        store=NullCampaignStore(),
        policy=AdmissionPolicy(max_concurrent_heavy=2, max_concurrent_light=8),
    )
    agent_a_scheduler = shared
    agent_b_scheduler = shared  # injected by factory.create() into both
    assert agent_a_scheduler is agent_b_scheduler

    async def run() -> None:
        # Agent A consumes both heavy slots.
        a1 = await agent_a_scheduler.acquire("vasp_tool", "heavy", None)
        a2 = await agent_a_scheduler.acquire("vasp_tool", "heavy", None)
        # Agent B's heavy acquire must block — same semaphore object.
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                agent_b_scheduler.acquire("vasp_tool", "heavy", None), timeout=0.1
            )
        agent_a_scheduler.release(a1)
        agent_a_scheduler.release(a2)

    asyncio.run(run())


# ── Test 5: submit_async queueing + drainer handoff ────────────────────────


def test_submit_async_queues_when_saturated():
    scheduler = ToolScheduler(
        store=NullCampaignStore(),
        policy=AdmissionPolicy(max_concurrent_heavy=1, max_concurrent_light=8),
    )

    async def run() -> None:
        scheduler.start()
        try:
            # First job grabs the only heavy slot and holds it until released.
            job1_gate = asyncio.Event()
            job1_done = asyncio.Event()

            async def job1() -> str:
                await job1_gate.wait()
                job1_done.set()
                return "job1-result"

            async def job2() -> str:
                return "job2-result"

            j1 = await scheduler.submit_async("vasp_tool", "heavy", None, job1)
            j2 = await scheduler.submit_async("vasp_tool", "heavy", None, job2)

            # Give the drainer a moment to admit job1 (the only slot).
            await asyncio.sleep(0.1)
            r1 = scheduler.get_job_status(j1)
            r2 = scheduler.get_job_status(j2)
            assert r1 is not None and r1.status == "running", r1.status
            assert r2 is not None and r2.status == "queued", r2.status

            # Release job1 → drainer admits job2.
            job1_gate.set()
            await asyncio.wait_for(job1_done.wait(), timeout=1.0)
            await asyncio.sleep(0.2)  # let drainer pick up job2

            r2b = scheduler.get_job_status(j2)
            assert r2b is not None
            assert r2b.status in ("running", "finished"), r2b.status
        finally:
            scheduler.stop()

    asyncio.run(run())


# ── Test 6: persist + recover across "restart" ─────────────────────────────


def test_recover_marks_running_orphaned_and_keeps_queued(tmp_path):
    db = tmp_path / "campaigns.sqlite"
    store = SqliteCampaignStore(db)

    # Phase 1: a scheduler crashes with one job running and one queued.
    s1 = ToolScheduler(store=store, policy=AdmissionPolicy(max_concurrent_heavy=2))
    j_running = "job_running_1"
    j_queued = "job_queued_1"
    from huginn.persistence.campaign import JobRecord

    store.upsert_job(
        JobRecord(
            job_id=j_running,
            tool_name="vasp_tool",
            status="running",
            cost_tier="heavy",
            started_at=time.time(),
        )
    )
    store.upsert_job(
        JobRecord(
            job_id=j_queued,
            tool_name="vasp_tool",
            status="queued",
            cost_tier="heavy",
            queue_position=0,
        )
    )

    # Phase 2: new process, new scheduler, same store → recover().
    s2 = ToolScheduler(store=store, policy=AdmissionPolicy(max_concurrent_heavy=2))
    summary = s2.recover()

    assert summary["orphaned"] == 1
    assert summary["requeued"] == 1
    running_after = store.get_job(j_running)
    queued_after = store.get_job(j_queued)
    assert running_after is not None
    assert running_after.status == "orphaned", running_after.status
    assert queued_after is not None
    assert queued_after.status == "queued", queued_after.status
    store.close()


# ── Test 7: main-path integration — concurrent acquire never exceeds cap ────


def test_concurrent_acquires_respect_heavy_cap():
    # Mirrors what _invoke_with_hooks does: N concurrent tool-call coroutines,
    # each acquire → do work → release. Asserts the heavy cap is never breached.
    scheduler = ToolScheduler(
        store=NullCampaignStore(),
        policy=AdmissionPolicy(max_concurrent_heavy=2, max_concurrent_light=8),
    )

    async def run() -> None:
        in_flight = 0
        peak = 0
        lock = asyncio.Lock()

        async def call() -> None:
            nonlocal in_flight, peak
            adm = await scheduler.acquire("vasp_tool", "heavy", None)
            try:
                async with lock:
                    in_flight += 1
                    peak = max(peak, in_flight)
                await asyncio.sleep(0.03)
                async with lock:
                    in_flight -= 1
            finally:
                scheduler.release(adm)

        await asyncio.gather(*(call() for _ in range(6)))
        assert peak <= 2, f"heavy cap breached: peak={peak}"
        # All 6 should have run (none starved).
        snap = scheduler.snapshot()
        assert snap.queued_depth == 0

    asyncio.run(run())


# ── Test 8: none-tier tools bypass the semaphore ───────────────────────────


def test_none_tier_unbounded():
    # cost_tier="none" tools (most tools) skip the semaphore entirely — they
    # must NOT block on heavy/light saturation. This is the transparency
    # guarantee that keeps the P1 regression net (autoloop/autoresearch mocks)
    # passing.
    scheduler = ToolScheduler(
        store=NullCampaignStore(),
        policy=AdmissionPolicy(max_concurrent_heavy=1, max_concurrent_light=1),
    )

    async def run() -> None:
        # Exhaust both heavy and light.
        h = await scheduler.acquire("vasp_tool", "heavy", None)
        la = await scheduler.acquire("structure_tool", "light", None)
        # A none-tier call must still go through instantly.
        adm = await asyncio.wait_for(
            scheduler.acquire("literature_tool", "none", None), timeout=0.2
        )
        assert adm.held_semaphore is False
        scheduler.release(adm)
        scheduler.release(h)
        scheduler.release(la)

    asyncio.run(run())
