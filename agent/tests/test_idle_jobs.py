"""I34: 空闲任务按租户隔离、有预算上限；超预算即停."""
from __future__ import annotations

from huginn.metacog.idle_runtime import IdleRuntime
from huginn.metacog.proactive_prep import (
    IdleJob,
    IdleJobQueue,
    PreparedContext,
    PreparedContextStore,
)


def _job(job_id: str, tenant: str, budget: int, state: str = "sleep") -> IdleJob:
    return IdleJob(
        job_id=job_id,
        tenant_id=tenant,
        persona_id="p1",
        job_type="demand_probe",
        budget_ms=budget,
        sleep_state=state,  # type: ignore[arg-type]
    )


def test_tenant_and_budget():
    queue = IdleJobQueue(per_tenant_max=10)
    assert queue.submit(_job("a1", "t1", 600), budget_cap_ms=1000) == "a1"
    assert queue.submit(_job("a2", "t1", 600), budget_cap_ms=1000) == "DROPPED"
    assert queue.submit(_job("b1", "t2", 600), budget_cap_ms=1000) == "b1"
    # 探针/预取在活跃间隙也能跑（睡眠门控只约束梦境巩固）
    assert queue.submit(_job("a3", "t1", 10, state="active")) == "a3"

    executed: list[str] = []
    counts_a = queue.run_due("t1", lambda job: executed.append(job.job_id),
                             budget_cap_ms=1000)
    assert counts_a == {"done": 2, "failed": 0, "dropped": 0}
    assert executed == ["a1", "a3"]
    assert queue.spent_ms("t1") == 610

    counts_b = queue.run_due("t2", lambda job: executed.append(job.job_id),
                             budget_cap_ms=1000)
    assert counts_b["done"] == 1
    assert queue.spent_ms("t2") == 600, "租户预算互不影响"
    assert executed == ["a1", "a3", "b1"]


def test_idle_runtime_runs_jobs_and_expires_candidates(tmp_path):
    store = PreparedContextStore(tmp_path / "prep.db")
    queue = IdleJobQueue()
    store.save(
        PreparedContext(
            prep_id="old",
            tenant_id="t1",
            persona_id="p1",
            predicted_need="stale",
            query_hint="q",
            ttl_s=1,
            created_at="2026-01-01T00:00:00+00:00",
        )
    )

    def _executor(job: IdleJob) -> None:
        from datetime import UTC, datetime

        store.save(
            PreparedContext(
                prep_id=f"prep:{job.job_id}",
                tenant_id="t1",
                persona_id="p1",
                predicted_need=job.job_type,
                query_hint="q",
                created_at=datetime.now(UTC).isoformat(),
            )
        )

    runtime = IdleRuntime(
        queue=queue,
        store=store,
        tenant_id="t1",
        jobs_fn=lambda: [
            IdleJob(
                job_id="j1", tenant_id="t1", persona_id="p1",
                job_type="demand_probe", budget_ms=10,
            )
        ],
        executor=_executor,
    )
    counts = runtime.run_once()
    assert counts["done"] == 1
    assert counts["candidates"] == 0, "过期候选在 tick 时已清理"
    assert store.get("old", tenant_id="t1")["status"] == "expired"
    assert store.get("prep:j1", tenant_id="t1")["status"] == "candidate"


def test_idle_runtime_disabled_does_not_start_thread():
    runtime = IdleRuntime(enabled=False)
    runtime.start()
    assert runtime._thread is None
    runtime.stop()


def test_queue_cap_per_tenant():
    queue = IdleJobQueue(per_tenant_max=1)
    assert queue.submit(_job("x1", "t1", 10)) == "x1"
    assert queue.submit(_job("x2", "t1", 10)) == "DROPPED"
