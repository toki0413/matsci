"""WakeScheduler — 轻量 scheduler tick, 消费 due wakes 并 resume sessions.

Portions derived from OpenWorker (https://github.com/andrewyng/openworker)
MIT License, Copyright (c) 2024 Andrew Ng
Source: coworker/automation/scheduler.py:1-113

策略 (跟 OpenWorker 一致):
- run-once-catch-up: 重启后第一次 tick 把所有 due wakes 触发一次 (不丢任务)
- skip-on-overlap: 同一 session 已有 run 在跑时跳过 (不堆栈)

不引入完整 TaskStore/ScheduledTask (OpenWorker 的 cron 系统). huginn 只需要
wake resume: tick 时扫 WakeStore.due(), 对每个 due wake 调 runner callback.

runner 签名: async def runner(session_id: str, wake: Wake) -> None
调用方注入 (e.g. AutoloopEngine 在 __init__ 时 spawn WakeScheduler, runner
调 self.run_once() resume session).

用法:
    from huginn.runtime.selfwake import get_wake_store
    from huginn.runtime.scheduler import WakeScheduler

    async def my_runner(session_id: str, wake) -> None:
        engine = engines.get(session_id)
        if engine:
            await engine.run_once()

    sched = WakeScheduler(get_wake_store(), my_runner, tick_seconds=30.0)
    sched.start()      # spawn asyncio task
    # ...
    await sched.stop()  # cancel + cleanup
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Optional

from huginn.runtime.selfwake import Wake, WakeStore

logger = logging.getLogger(__name__)

Runner = Callable[[str, Wake], Awaitable[None]]


class WakeScheduler:
    """轻量 wake scheduler. tick 时扫 WakeStore.due(), spawn runner for each.

    单 asyncio.Task 跑 _loop, 内部调 _tick. 每个 due wake 用
    asyncio.create_task 独立 spawn (不阻塞 loop), skip-on-overlap 用
    _running_sessions set 防 same-session 堆栈.

    ponytail: 不引入 cron 表达式 (OpenWorker 也没有). 只支持 interval tick +
    event trigger (complete_job/fire_event 由外部调 WakeStore). 升级路径:
    接 APScheduler 或引入 ScheduledTask 模型.
    """

    def __init__(
        self,
        store: WakeStore,
        runner: Runner,
        *,
        tick_seconds: float = 30.0,
    ) -> None:
        self.store = store
        self.runner = runner
        self.tick_seconds = tick_seconds
        self._task: Optional[asyncio.Task] = None
        self._running_sessions: set[str] = set()  # skip-on-overlap
        self._spawned: set[asyncio.Task] = set()  # 防止 GC 回收 spawned runs

    def start(self) -> None:
        """spawn scheduler loop as asyncio.Task. 幂等 (重复调只 spawn 一次)."""
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        """cancel scheduler loop + 所有 spawned runs. 幂等."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        # spawned runs 不能比 scheduler 活更久 (一个挂起的 run 不应跑过 scheduler)
        for spawned in list(self._spawned):
            spawned.cancel()
            try:
                await spawned
            except asyncio.CancelledError:
                pass
        self._spawned.clear()
        self._running_sessions.clear()

    async def _loop(self) -> None:
        """First pass = catch-up (重启时触发所有 due wakes), 然后定期 tick."""
        # run-once-catch-up: 重启后第一次 tick 把错过的 due 都触发
        try:
            await self._tick(trigger="catchup")
        except Exception:
            logger.exception("wake scheduler catch-up failed")
        while True:
            await asyncio.sleep(self.tick_seconds)
            try:
                await self._tick(trigger="schedule")
            except Exception:
                logger.exception("wake scheduler tick failed")

    async def _tick(self, *, trigger: str) -> None:
        """扫 due wakes, 每个 spawn 一个独立 task (不阻塞 loop)."""
        due = self.store.due()
        for wake in due:
            # skip-on-overlap: 同 session 已有 run 在跑 → 跳过
            if wake.session_id in self._running_sessions:
                logger.info(
                    "skip wake %s — session %s still running",
                    wake.id, wake.session_id,
                )
                continue
            # spawn 前就占 session, 防止同 tick 内同 session 的其他 wake 也被 spawn
            self._running_sessions.add(wake.session_id)
            spawned = asyncio.create_task(self._run_one(wake, trigger=trigger))
            self._spawned.add(spawned)
            spawned.add_done_callback(self._spawned.discard)

    async def _run_one(self, wake: Wake, *, trigger: str) -> None:
        """执行单个 wake: 调 runner, 完成后 mark_fired + 释放 session 占用."""
        try:
            await self.runner(wake.session_id, wake)
        except Exception:
            logger.exception(
                "wake %s (session %s) runner failed [trigger=%s]",
                wake.id, wake.session_id, trigger,
            )
        finally:
            self.store.mark_fired(wake.id)
            self._running_sessions.discard(wake.session_id)


if __name__ == "__main__":
    import shutil
    import tempfile as _tf
    from datetime import datetime, timedelta, timezone
    from pathlib import Path

    async def _selfcheck():
        ws = Path(_tf.mkdtemp(prefix="huginn_sched_test_"))
        try:
            store = WakeStore(ws / "wakes.json")

            # 1. 基本: past timer → tick → runner 被调 + mark_fired
            runs: list[tuple[str, str]] = []  # (session_id, wake_id)
            async def runner(session_id: str, wake: Wake) -> None:
                runs.append((session_id, wake.id))

            sched = WakeScheduler(store, runner, tick_seconds=0.05)
            past = datetime.now(timezone.utc) - timedelta(seconds=5)
            w1 = store.add_timer("s1", past)
            sched.start()
            # 等 catchup tick 跑完
            await asyncio.sleep(0.2)
            assert len(runs) == 1, f"expected 1 run, got {len(runs)}"
            assert runs[0] == ("s1", w1.id)
            assert store.get(w1.id).state == "fired"
            await sched.stop()
            print("1. catchup tick → runner + mark_fired OK")

            # 2. skip-on-overlap: 同 session 的第二个 wake 在第一个还在跑时跳过
            runs.clear()
            started = asyncio.Event()
            release = asyncio.Event()
            async def slow_runner(session_id: str, wake: Wake) -> None:
                runs.append((session_id, wake.id))
                started.set()
                await release.wait()  # 模拟长 run

            store2 = WakeStore(ws / "wakes2.json")
            sched2 = WakeScheduler(store2, slow_runner, tick_seconds=0.05)
            past2 = datetime.now(timezone.utc) - timedelta(seconds=5)
            w_a = store2.add_timer("s_slow", past2)
            w_b = store2.add_timer("s_slow", past2)  # 同 session, 同时 due
            sched2.start()
            await asyncio.sleep(0.2)
            # 第一个 wake 应已开始 (slow_runner 阻塞在 release.wait)
            assert started.is_set(), "first wake should have started"
            assert len(runs) == 1, f"only 1 should run, got {len(runs)}"
            # 第二个 wake 应被 skip-on-overlap 跳过
            assert store2.get(w_b.id).state != "fired", "second wake should be skipped"
            # 释放第一个, 让它完成
            release.set()
            await asyncio.sleep(0.2)
            assert store2.get(w_a.id).state == "fired"
            # 第二个 wake 在下一个 tick 会被 pickup (第一个已释放)
            await asyncio.sleep(0.2)
            assert len(runs) == 2, f"second wake should run after first releases, got {len(runs)}"
            assert store2.get(w_b.id).state == "fired"
            await sched2.stop()
            print("2. skip-on-overlap + release → resume OK")

            # 3. runner 异常不阻塞其他 wake
            runs.clear()
            async def failing_runner(session_id: str, wake: Wake) -> None:
                if session_id == "s_fail":
                    raise RuntimeError("intentional failure")
                runs.append((session_id, wake.id))

            store3 = WakeStore(ws / "wakes3.json")
            sched3 = WakeScheduler(store3, failing_runner, tick_seconds=0.05)
            past3 = datetime.now(timezone.utc) - timedelta(seconds=5)
            w_fail = store3.add_timer("s_fail", past3)
            w_ok = store3.add_timer("s_ok", past3)
            sched3.start()
            await asyncio.sleep(0.2)
            # 失败的 wake 仍 mark_fired, 不阻塞 ok wake
            assert store3.get(w_fail.id).state == "fired"
            assert store3.get(w_ok.id).state == "fired"
            assert len(runs) == 1
            assert runs[0] == ("s_ok", w_ok.id)
            await sched3.stop()
            print("3. runner exception → other wake still runs OK")

            # 4. stop 幂等 + 不残留 task
            sched4 = WakeScheduler(WakeStore(), runner, tick_seconds=1.0)
            sched4.start()
            assert sched4._task is not None
            await sched4.stop()
            assert sched4._task is None
            # 二次 stop 不报错
            await sched4.stop()
            print("4. stop idempotent OK")

            # 5. start 幂等 (重复调只 spawn 一次)
            sched5 = WakeScheduler(WakeStore(), runner, tick_seconds=1.0)
            sched5.start()
            t1 = sched5._task
            sched5.start()
            assert sched5._task is t1, "second start should not spawn new task"
            await sched5.stop()
            print("5. start idempotent OK")

            print("ALL CHECKS PASSED")
        finally:
            shutil.rmtree(ws, ignore_errors=True)

    asyncio.run(_selfcheck())
