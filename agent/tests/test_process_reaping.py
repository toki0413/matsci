"""退出兜底回收回归 — 堵住 PPID→1 孤儿泄漏 (run87).

背景: 沙箱子进程以 ``start_new_session`` / ``CREATE_NEW_PROCESS_GROUP`` 自立新组
执行, 这修好了"超时杀死直接子进程、孙进程 orphan"的问题, 但同时带来另一条泄漏:
**父进程先退出**时子进程收不到任何信号, 直接成为 PPID→1 的孤儿继续全速运行
(run87 实测: 主进程退出后实验子进程又跑了 ~15min).

修法: 运行中的子进程登记进 ``huginn.utils.process`` 的在册表, 退出前
``kill_tracked_children()`` 按进程组整组回收. 本测试覆盖: 登记/注销、整组回收、
SandboxExecutor 的登记生命周期、ToolScheduler.stop 取消在飞任务.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from huginn.utils import process as proc_util


def _spawn_sleeper():
    """起一个自立新组的 sleep 子进程 (与沙箱同一进程组语义)."""
    import subprocess
    import sys

    kwargs, own_group = proc_util.new_group_popen_kwargs()
    p = subprocess.Popen(  # noqa: S603 — 测试自有命令
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **kwargs,
    )
    proc_util.track_live_child(p.pid, own_group=own_group)
    return p


def _is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def test_kill_tracked_children_reaps_process_group():
    """在册子进程被整组回收; 回收后登记表清空, 进程确实已死."""
    p = _spawn_sleeper()
    try:
        assert _is_alive(p.pid)
        killed = proc_util.kill_tracked_children()
        assert killed >= 1
        # 等 SIGKILL 落地 (僵尸在父进程 wait 前仍占 pid, 但已不再运行).
        p.wait(timeout=10)
        assert p.returncode != 0
    finally:
        if p.poll() is None:
            p.kill()
            p.wait(timeout=5)


def test_untracked_child_survives_reaping():
    """已注销的子进程不被退出兜底误杀 (防 pid 复用)."""
    p = _spawn_sleeper()
    try:
        proc_util.untrack_live_child(p.pid)
        proc_util.kill_tracked_children()
        assert p.poll() is None, "注销后的子进程不应被回收"
    finally:
        p.kill()
        p.wait(timeout=5)


def test_kill_tracked_children_is_idempotent():
    """重复调用安全: 第二次无在册项, 返回 0 且不抛."""
    p = _spawn_sleeper()
    try:
        assert proc_util.kill_tracked_children() >= 1
        assert proc_util.kill_tracked_children() == 0
        p.wait(timeout=10)
    finally:
        if p.poll() is None:
            p.kill()
            p.wait(timeout=5)


@pytest.mark.skipif(os.name != "posix", reason="POSIX 进程组语义")
def test_sandbox_executor_tracks_then_untracks(tmp_path: Path):
    """SandboxExecutor 运行期间登记子进程, 结束后注销 (不残留登记项)."""
    import sys

    from huginn.security.sandbox import SandboxConfig, SandboxExecutor

    cfg = SandboxConfig(
        allowed_executables={"python3", "python", "echo"},
        allowed_work_dirs={tmp_path},
        strict_work_dir=True,
    )
    ex = SandboxExecutor(config=cfg)

    assert not proc_util._LIVE_CHILDREN, "前置: 登记表应为空"

    result = ex.run(
        [sys.executable, "-c", "print('ok')"],
        cwd=tmp_path,
        timeout=30,
    )
    assert result.returncode == 0
    assert "ok" in result.stdout
    assert not proc_util._LIVE_CHILDREN, "正常结束后登记表应清空"


@pytest.mark.skipif(os.name != "posix", reason="POSIX 进程组语义")
def test_sandbox_executor_untracks_on_timeout(tmp_path: Path):
    """超时路径: 整组回收后同样注销登记, 不留残项."""
    import sys

    from huginn.security.sandbox import SandboxConfig, SandboxExecutor

    cfg = SandboxConfig(
        allowed_executables={"python3", "python"},
        allowed_work_dirs={tmp_path},
        strict_work_dir=True,
    )
    ex = SandboxExecutor(config=cfg)
    result = ex.run(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        cwd=tmp_path,
        timeout=1,
    )
    assert result.timed_out
    assert not proc_util._LIVE_CHILDREN, "超时回收后登记表应清空"


def test_scheduler_stop_cancels_live_tasks(tmp_path: Path):
    """ToolScheduler.stop 只停 drainer 不够 —— 在飞任务必须一并取消."""
    from huginn.scheduling.scheduler import ToolScheduler

    async def _scenario():
        sched = ToolScheduler()
        started = asyncio.Event()

        async def _job():
            started.set()
            await asyncio.sleep(30)  # 模拟长跑后台任务
            return "done"

        sched.start()
        await sched.submit_async("bash_tool", "heavy", None, _job)
        await asyncio.wait_for(started.wait(), timeout=5)
        await asyncio.sleep(0.05)
        live = list(sched._live_tasks.values())
        assert live and not live[0].done()

        sched.stop()
        await asyncio.sleep(0.05)
        assert all(t.cancelled() or t.done() for t in live)

    asyncio.run(_scenario())
