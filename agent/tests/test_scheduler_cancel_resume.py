"""ToolScheduler 真取消恢复 + 后台 bash 语义的回归测试.

覆盖 (d) 真 cancel 语义、resume orphaned、以及 job_tool.cancel 真实接线到
scheduler. 运行中作业用短暂的 asyncio.sleep 模拟, 不依赖 pytest-asyncio,
沿用 test_tool_scheduler.py 的 asyncio.run 模式.
"""

from __future__ import annotations

import asyncio

from huginn.persistence.campaign import (
    JobRecord,
    NullCampaignStore,
    SqliteCampaignStore,
)
from huginn.scheduling import AdmissionPolicy, ToolScheduler
from huginn.tools.job_tool import JobTool, JobToolInput

# ── (a) queued 作业 cancel: 成功且状态 = cancelled ────────────────────────


def test_cancel_queued_job():
    scheduler = ToolScheduler(
        store=NullCampaignStore(),
        policy=AdmissionPolicy(max_concurrent_heavy=1, max_concurrent_light=8),
    )

    async def run():
        scheduler.start()
        gate = asyncio.Event()
        ran: list[str] = []

        async def blocker():
            await gate.wait()
            ran.append("blocker")

        async def target():
            ran.append("target")
            return "t"

        j1 = await scheduler.submit_async("bash_tool", "heavy", None, blocker)
        j2 = await scheduler.submit_async("bash_tool", "heavy", None, target)

        # drainer 让 blocker 占住唯一 heavy 槽位, target 排队
        for _ in range(100):
            rec1 = scheduler.get_job_status(j1)
            if rec1 is not None and rec1.status == "running":
                break
            await asyncio.sleep(0.01)
        assert scheduler.get_job_status(j2).status == "queued"

        cancelled = await scheduler.cancel(j2)
        assert cancelled
        rec2 = scheduler.get_job_status(j2)
        assert rec2 is not None and rec2.status == "cancelled", rec2.status

        # 释放槽位后 target 不应再被 drainer 吸收执行
        gate.set()
        await asyncio.sleep(0.05)
        assert "target" not in ran
        scheduler.stop()

    asyncio.run(run())


# ── (b) 运行中作业 cancel: 底层 asyncio.Task 被取消 ───────────────────────


def test_cancel_running_task():
    scheduler = ToolScheduler(
        store=NullCampaignStore(),
        policy=AdmissionPolicy(max_concurrent_heavy=2, max_concurrent_light=8),
    )

    async def long_task():
        while True:
            await asyncio.sleep(0.02)

    async def run():
        scheduler.start()
        job_id = await scheduler.submit_async("bash_tool", "heavy", None, long_task)

        # 等 drainer 吸收为 running
        for _ in range(100):
            rec = scheduler.get_job_status(job_id)
            if rec is not None and rec.status == "running":
                break
            await asyncio.sleep(0.01)
        assert scheduler.get_job_status(job_id).status == "running"

        task = scheduler._live_tasks[job_id]
        ok = await scheduler.cancel(job_id)
        assert ok
        await asyncio.sleep(0.05)  # 让取消在事件循环里传播
        assert task.cancelled(), "底层 asyncio.Task 应被取消"
        rec = scheduler.get_job_status(job_id)
        assert rec is not None and rec.status == "cancelled", rec.status
        scheduler.stop()

    asyncio.run(run())


# ── (c) recover/redeploy 后 orphaned 作业可 resume ───────────────────────


def _test_resume_orphaned_impl(tmp_path):
    db = tmp_path / "campaigns.sqlite"
    store = SqliteCampaignStore(db)

    # Phase 1: "崩溃"遗留一个 running 作业
    store.upsert_job(
        JobRecord(
            job_id="job_x",
            tool_name="bash_tool",
            status="running",
            cost_tier="heavy",
            queue_position=0,
            cores_requested=1.0,
        )
    )

    # Phase 2: 新进程, 同一 store, 新 scheduler
    scheduler = ToolScheduler(
        store=store, policy=AdmissionPolicy(max_concurrent_heavy=2)
    )

    async def run():
        scheduler.start()
        summary = scheduler.recover()
        assert summary["orphaned"] == 1
        assert store.get_job("job_x").status == "orphaned"

        executed: list[bool] = []

        async def factory():
            executed.append(True)
            return "done"

        def provider(job_id, rec):
            return factory

        resumed = scheduler.resume_orphaned(factory_provider=provider)
        assert resumed == 1
        assert store.get_job("job_x").status == "queued", "resume 应先重新排队"

        # drainer 吸收并跑到 finished
        for _ in range(200):
            rec = store.get_job("job_x")
            if rec is not None and rec.status == "finished":
                break
            await asyncio.sleep(0.01)
        assert store.get_job("job_x").status == "finished"
        assert executed == [True], "恢复后的 factory 应当被真实执行"
        scheduler.stop()

    asyncio.run(run())
    store.close()


def test_resume_orphaned_after_restart(tmp_path):
    _test_resume_orphaned_impl(tmp_path)


def test_resume_uses_recorded_factory():
    """resume_orphaned 无 factory_provider 时复用 submit_async 记录的 factory."""
    scheduler = ToolScheduler(
        store=NullCampaignStore(),
        policy=AdmissionPolicy(max_concurrent_heavy=2),
    )

    async def factory():
        return "ok"

    async def run():
        # 先不启动 drainer, 避免 submit 后队列被立即吸收, 制造确定性的 orphaned 语义.
        job_id = await scheduler.submit_async("bash_tool", "heavy", None, factory)
        # 摘掉内存队列条目: 模拟崩溃遗留的未运行作业 (持久层仍为 queued)
        for i, q in enumerate(scheduler._queue):
            if q.job_id == job_id:
                del scheduler._queue[i]
                break
        assert job_id in scheduler._factories, "submit_async 应记录原始 factory"
        rec = scheduler.store.get_job(job_id)
        rec.status = "orphaned"
        scheduler.store.upsert_job(rec)

        scheduler.start()
        # 不传 factory_provider → 应回退到 _factories 里记录的原始 factory
        assert scheduler.resume_orphaned() == 1, "复用记录的 factory 恢复成功"
        assert scheduler.store.get_job(job_id).status == "queued"

        # drainer 吸收并跑完
        for _ in range(200):
            r = scheduler.store.get_job(job_id)
            if r is not None and r.status == "finished":
                break
            await asyncio.sleep(0.01)
        assert scheduler.store.get_job(job_id).status == "finished"
        scheduler.stop()

    asyncio.run(run())


# ── (d) job_tool.cancel 真实调用 scheduler ──────────────────────────────


def test_job_tool_cancel_calls_scheduler():
    scheduler = ToolScheduler(
        store=NullCampaignStore(),
        policy=AdmissionPolicy(max_concurrent_heavy=1),
    )
    tool = JobTool(scheduler=scheduler)

    async def run():
        async def dummy():
            return "x"

        await scheduler.submit_async("bash_tool", "heavy", None, dummy)
        job_id = scheduler.store.list_queued_jobs()[0].job_id

        result = await tool.call(JobToolInput(action="cancel", job_id=job_id), None)
        assert result.success, result.error
        assert result.data["status"] == "cancelled"
        rec = scheduler.get_job_status(job_id)
        assert rec is not None and rec.status == "cancelled", rec.status

    asyncio.run(run())


def test_job_tool_cancel_without_scheduler_errors():
    """未连接 scheduler 时 cancel 返回明确错误, 而非假成功."""
    tool = JobTool()  # 未注入 scheduler, context 也为 None

    async def run():
        result = await tool.call(JobToolInput(action="cancel", job_id="job_x"), None)
        assert not result.success
        assert result.error is not None and "scheduler not connected" in result.error

    asyncio.run(run())
