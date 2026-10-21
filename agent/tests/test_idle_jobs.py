"""I34: 空闲任务按租户隔离、有预算上限；超预算即停."""
from __future__ import annotations

from huginn.metacog.proactive_prep import IdleJob, IdleJobQueue


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
    assert queue.submit(_job("a3", "t1", 10, state="active")) == "DROPPED"

    executed: list[str] = []
    counts_a = queue.run_due("t1", lambda job: executed.append(job.job_id),
                             budget_cap_ms=1000)
    assert counts_a == {"done": 1, "failed": 0, "dropped": 0}
    assert executed == ["a1"]
    assert queue.spent_ms("t1") == 600

    counts_b = queue.run_due("t2", lambda job: executed.append(job.job_id),
                             budget_cap_ms=1000)
    assert counts_b["done"] == 1
    assert queue.spent_ms("t2") == 600, "租户预算互不影响"
    assert executed == ["a1", "b1"]


def test_queue_cap_per_tenant():
    queue = IdleJobQueue(per_tenant_max=1)
    assert queue.submit(_job("x1", "t1", 10)) == "x1"
    assert queue.submit(_job("x2", "t1", 10)) == "DROPPED"
