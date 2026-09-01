"""方案A M4: 安全执行闸门 (权威仲裁 + 死手/心跳) 接线测试."""

from __future__ import annotations

import pytest

from huginn.security.control_authority import AUTONOMOUS, LOCAL, PHYSICAL, REMOTE
from huginn.security.control_safety import (
    CommandDeniedError,
    ExecutionGuard,
    SafetyStopError,
)


def _mk_clock():
    now = [0.0]
    return now, lambda: now[0]


def test_authority_denies_lower_priority_and_preempts():
    """权威仲裁: 低优先级持权后拒更高冲突; 物理/本地可抢占远程/自治."""
    now, clock = _mk_clock()
    guard = ExecutionGuard(max_idle=5.0, clock=clock)
    seen = []

    guard.command(REMOTE, lambda: seen.append("remote"))
    assert seen == ["remote"]

    # AUTONOMOUS(rank3) 低于当前 owner REMOTE(rank2) → 拒绝.
    with pytest.raises(CommandDeniedError):
        guard.command(AUTONOMOUS, lambda: seen.append("auto"))

    # PHYSICAL(rank0) 可抢占 REMOTE → 执行.
    guard.command(PHYSICAL, lambda: seen.append("physical"))
    assert seen == ["remote", "physical"]
    assert guard.authority.owner() == "physical"

    # LOCAL(rank1) 低于当前 owner PHYSICAL → 拒绝.
    with pytest.raises(CommandDeniedError):
        guard.command(LOCAL, lambda: seen.append("local"))


def test_deadman_latches_after_idle_and_reset_resumes():
    """死手: 命令停止超时 → 安全停机拒绝; reset 后恢复并重新喂心跳."""
    now, clock = _mk_clock()
    safe_stops = []
    guard = ExecutionGuard(5.0, clock=clock, on_safe_stop=lambda: safe_stops.append(1))

    guard.command(LOCAL, lambda: None)  # now=0, 心跳
    now[0] = 2.0
    guard.command(LOCAL, lambda: None)  # 仍在窗口内
    assert guard.monitor() is False
    assert safe_stops == []

    now[0] = 8.0  # 距上次命令 6s > 5s 超时
    assert guard.monitor() is True
    assert safe_stops == [1]  # on_safe_stop 触发一次

    # 已停机 → 命令被拒.
    with pytest.raises(SafetyStopError):
        guard.command(LOCAL, lambda: None)

    # 人工接管后 reset → 恢复执行并重新喂心跳.
    guard.reset()
    now[0] = 9.0
    guard.command(LOCAL, lambda: None)
    now[0] = 10.0  # 9→10 在 5s 窗口内
    assert guard.monitor() is False


def test_safety_guard_wraps_an_executor_call():
    """接线: 用闸门包一层执行器调用 (如 workspace.execute) 不破坏其返回."""
    now, clock = _mk_clock()
    guard = ExecutionGuard(5.0, clock=clock)

    def compute():
        return 42

    assert guard.command(LOCAL, compute) == 42
    assert guard.command(LOCAL, compute) == 42  # 当前 owner 重复命令不被误拒
    with pytest.raises(CommandDeniedError):
        guard.command(REMOTE, compute)  # REMOTE(rank2) 低于 LOCAL(rank1) 持权 → 拒
