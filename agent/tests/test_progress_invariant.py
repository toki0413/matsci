"""P3.2 在线进展不变量测试 — 执行前阶段打转 ⇒ 强制推进流水线.

覆盖 progress_invariant_action 的纯函数契约:
  1. 窗口不足 → 不触发
  2. 尾部含 execute/validate/learn/pivot → 不触发 (在前进)
  3. 纯 skip/observe (monitor-hold) → 不触发
  4. 纯 hypothesize 打转且有假设 → 强制 plan
  5. 纯 hypothesize 打转且无假设 → 强制 hypothesize
  6. 尾部含 plan 但无 execute → 强制 execute
"""

from __future__ import annotations

from huginn.autoloop.cognitive_checks import progress_invariant_action


def test_below_window_no_trigger():
    assert progress_invariant_action({"hypothesis": "h"}, ["hypothesize", "hypothesize"]) is None


def test_execute_in_tail_no_trigger():
    hist = ["hypothesize", "plan", "execute", "validate"]
    assert progress_invariant_action({"hypothesis": "h", "plan": {}}, hist) is None


def test_pivot_in_tail_no_trigger():
    hist = ["hypothesize", "hypothesize", "pivot", "observe"]
    assert progress_invariant_action({"hypothesis": "h"}, hist) is None


def test_pure_skip_monitor_hold_no_trigger():
    hist = ["skip", "skip", "skip", "skip"]
    assert progress_invariant_action({}, hist) is None


def test_pure_observe_no_trigger():
    hist = ["observe", "observe", "observe", "observe"]
    assert progress_invariant_action({"hypothesis": "h"}, hist) is None


def test_hypothesize_loop_forces_plan():
    hist = ["hypothesize", "hypothesize", "hypothesize", "hypothesize"]
    assert progress_invariant_action({"hypothesis": "h"}, hist) == "plan"


def test_hypothesize_loop_without_hypothesis_forces_hypothesize():
    hist = ["observe", "hypothesize", "hypothesize", "hypothesize"]
    assert progress_invariant_action({"hypothesis": ""}, hist) == "hypothesize"


def test_plan_without_execute_forces_execute():
    hist = ["hypothesize", "plan", "hypothesize", "plan"]
    assert progress_invariant_action({"hypothesis": "h", "plan": {"mode": "coder"}}, hist) == "execute"


def test_window_env_override():
    # window=2 时两轮 hypothesize 即触发
    hist = ["observe", "hypothesize", "hypothesize"]
    assert progress_invariant_action({"hypothesis": "h"}, hist, window=2) == "plan"
    # window=4 时同样历史不足 → 不触发
    assert progress_invariant_action({"hypothesis": "h"}, hist, window=4) is None
