"""monitor-hold 减负策略单测 — 连续无观测跳过高成本推理 (对齐 Hermes monitor)."""

from __future__ import annotations

from huginn.autoloop.monitor_hold import (
    HOLD_THRESHOLD,
    MonitorHoldState,
    decide_hold,
)


def test_activity_resets_streak_and_never_holds():
    """有新活动 → 不 hold 且重置连续空计数."""
    st = MonitorHoldState(empty_streak=5)
    d = decide_hold(st, had_activity=True)
    assert d.hold is False
    assert d.empty_streak == 0
    assert st.last_had_activity is True


def test_below_threshold_gives_one_chance():
    """刚安静 (低于阈值) → 给一次处理机会, 不立即 hold."""
    st = MonitorHoldState()
    d = decide_hold(st, had_activity=False)
    assert d.hold is False
    assert d.empty_streak == 1
    assert d.reason == f"quiet for {1}x (below {HOLD_THRESHOLD}), give one processing chance"


def test_quiet_reaching_threshold_holds():
    """连续空观测达阈值 → 本回合 hold, 跳过 LLM."""
    st = MonitorHoldState()
    for _ in range(HOLD_THRESHOLD - 1):
        decide_hold(st, had_activity=False)
    d = decide_hold(st, had_activity=False)
    assert d.empty_streak == HOLD_THRESHOLD
    assert d.hold is True


def test_quiet_reaching_threshold_holds_clean():
    """连续空观测达到阈值 → hold (干净、无多余占位)."""
    st = MonitorHoldState()
    for _ in range(HOLD_THRESHOLD - 1):
        decide_hold(st, had_activity=False)
    d = decide_hold(st, had_activity=False)
    assert d.empty_streak == HOLD_THRESHOLD
    assert d.hold is True
    assert "skip expensive reasoning" in d.reason


def test_hold_then_activity_recovers():
    """hold 后一旦有新活动 → 立即恢复, 不再 hold."""
    st = MonitorHoldState()
    for _ in range(HOLD_THRESHOLD):
        decide_hold(st, had_activity=False)
    assert decide_hold(st, had_activity=False).hold is True
    # 恢复
    d = decide_hold(st, had_activity=True)
    assert d.hold is False
    assert st.empty_streak == 0
