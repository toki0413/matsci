"""DecisionArbiter — 统一决策仲裁层测试.

覆盖:
  - 8 条仲裁优先级 (gate pass+stop / 预算耗尽 / block / gaps_hint / pass+switch / 等)
  - build_context 从 bandit / gate_decision 收集信号
  - Decision.should_stop 属性
  - 异常 fallback (bandit policy 抛异常 → continue)
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from huginn.metacog.decision_arbiter import (
    Decision,
    DecisionArbiter,
    DecisionContext,
)

# ── Decision dataclass ─────────────────────────────────────────────


def test_decision_should_stop_true_when_action_stop():
    d = Decision(action="stop")
    assert d.should_stop is True


def test_decision_should_stop_false_otherwise():
    for action in ("continue", "switch_tool", "requery", "explore"):
        assert Decision(action=action).should_stop is False


def test_decision_default_fields():
    d = Decision(action="continue")
    assert d.reason == ""
    assert d.source == ""
    assert d.confidence == 1.0


# ── DecisionContext defaults ───────────────────────────────────────


def test_context_defaults():
    c = DecisionContext()
    assert c.csm_state == ""
    assert c.bandit_advice == "continue"
    assert c.gate_status == "pending"
    assert c.gate_should_stop is False
    assert c.active_signals == []
    assert c.last_errors == []


def test_context_lists_are_independent():
    a = DecisionContext()
    b = DecisionContext()
    a.active_signals.append("x")
    a.last_errors.append("y")
    assert b.active_signals == []
    assert b.last_errors == []


# ── 仲裁优先级 1-8 ─────────────────────────────────────────────────


@pytest.fixture
def arbiter():
    return DecisionArbiter()


def test_priority_1_gate_pass_and_should_stop(arbiter):
    """Gate 判定完成 → stop."""
    ctx = DecisionContext(gate_status="pass", gate_should_stop=True)
    d = arbiter.evaluate(ctx)
    assert d.action == "stop"
    assert d.source == "gate"


def test_priority_2_budget_exhausted(arbiter):
    """预算耗尽 → stop (即便 gate 未 pass)."""
    ctx = DecisionContext(iteration=10, max_iterations=10, gate_status="pending")
    d = arbiter.evaluate(ctx)
    assert d.action == "stop"
    assert d.source == "budget_exhausted"


def test_priority_3_gate_block_continue(arbiter):
    """Gate block + bandit continue → continue."""
    ctx = DecisionContext(
        gate_status="block", gate_reason="effort low", bandit_advice="continue"
    )
    d = arbiter.evaluate(ctx)
    assert d.action == "continue"
    assert "Gate blocked" in d.reason


def test_priority_3_gate_block_switch(arbiter):
    """Gate block + bandit switch → switch_tool."""
    ctx = DecisionContext(gate_status="block", bandit_advice="switch")
    d = arbiter.evaluate(ctx)
    assert d.action == "switch_tool"
    assert d.source == "arbiter"


def test_priority_4_gaps_hint(arbiter):
    """Gate gaps_hint → continue (gap 作为 hint)."""
    ctx = DecisionContext(gate_status="gaps_hint", gate_reason="missing X")
    d = arbiter.evaluate(ctx)
    assert d.action == "continue"
    assert "Gaps" in d.reason


def test_priority_5_gate_pass_bandit_continue(arbiter):
    ctx = DecisionContext(gate_status="pass", bandit_advice="continue")
    d = arbiter.evaluate(ctx)
    assert d.action == "continue"
    assert d.source == "arbiter"


def test_priority_5_gate_pass_bandit_switch(arbiter):
    ctx = DecisionContext(gate_status="pass", bandit_advice="switch")
    d = arbiter.evaluate(ctx)
    assert d.action == "switch_tool"
    assert d.source == "bandit"


def test_priority_6_gate_pass_bandit_requery(arbiter):
    """冲突: Gate pass + bandit requery → arbiter 允许 requery."""
    ctx = DecisionContext(gate_status="pass", bandit_advice="requery")
    d = arbiter.evaluate(ctx)
    assert d.action == "requery"
    assert d.source == "arbiter"


def test_priority_7_gate_pending_bandit_switch(arbiter):
    ctx = DecisionContext(gate_status="pending", bandit_advice="switch")
    d = arbiter.evaluate(ctx)
    assert d.action == "switch_tool"


def test_priority_7_gate_pending_bandit_requery(arbiter):
    ctx = DecisionContext(gate_status="pending", bandit_advice="requery")
    d = arbiter.evaluate(ctx)
    assert d.action == "requery"


def test_priority_8_default_continue(arbiter):
    ctx = DecisionContext(gate_status="pending", bandit_advice="continue")
    d = arbiter.evaluate(ctx)
    assert d.action == "continue"
    assert d.source == "arbiter"


# ── build_context ──────────────────────────────────────────────────


def test_build_context_collects_bandit_advice(arbiter):
    class _MockBandit:
        def policy(self): return "switch"
    ctx = arbiter.build_context(bandit=_MockBandit())
    assert ctx.bandit_advice == "switch"


def test_build_context_collects_gate_decision(arbiter):
    gate = SimpleNamespace(status="pass", should_stop=True, reason="ok")
    ctx = arbiter.build_context(gate_decision=gate)
    assert ctx.gate_status == "pass"
    assert ctx.gate_should_stop is True
    assert ctx.gate_reason == "ok"


def test_build_context_bandit_failure_defaults_continue(arbiter):
    """bandit.policy() 抛异常 → bandit_advice fallback 到 continue."""
    class _BrokenBandit:
        def policy(self): raise RuntimeError("boom")
    ctx = arbiter.build_context(bandit=_BrokenBandit())
    assert ctx.bandit_advice == "continue"


def test_build_context_none_inputs_safe(arbiter):
    """bandit/gate 都是 None 时不抛."""
    ctx = arbiter.build_context(bandit=None, gate_decision=None)
    assert ctx.bandit_advice == "continue"
    assert ctx.gate_status == "pending"


def test_build_context_passes_runtime_fields(arbiter):
    ctx = arbiter.build_context(
        csm_state="S4_CONSTRUCT",
        iteration=3,
        max_iterations=10,
        turns_count=5,
        tool_calls_count=20,
        active_signals=["low_surprise"],
        last_errors=["tool_x_failed"],
    )
    assert ctx.csm_state == "S4_CONSTRUCT"
    assert ctx.iteration == 3
    assert ctx.max_iterations == 10
    assert ctx.turns_count == 5
    assert ctx.tool_calls_count == 20
    assert ctx.active_signals == ["low_surprise"]
    assert ctx.last_errors == ["tool_x_failed"]


# ── 边界场景 ──────────────────────────────────────────────────────


def test_budget_zero_max_iterations_does_not_trigger_stop(arbiter):
    """max_iterations=0 时不算预算耗尽 (avoid 0/0 误判)."""
    ctx = DecisionContext(iteration=0, max_iterations=0)
    d = arbiter.evaluate(ctx)
    # max=0 跳过预算检查 → 走默认 continue
    assert d.action == "continue"


def test_budget_one_off(arbiter):
    """iteration=9, max=10 → 不算耗尽, 走默认 continue."""
    ctx = DecisionContext(iteration=9, max_iterations=10, gate_status="pending")
    d = arbiter.evaluate(ctx)
    assert d.action == "continue"


def test_gate_pass_no_should_stop_does_not_stop(arbiter):
    """Gate pass 但 should_stop=False → 不应 stop (例如还有更优工具要试)."""
    ctx = DecisionContext(gate_status="pass", gate_should_stop=False, bandit_advice="continue")
    d = arbiter.evaluate(ctx)
    assert d.action != "stop"


def test_gate_block_overrides_budget_exhausted_priority(arbiter):
    """优先级 2 (预算耗尽) 在 1 (gate pass+stop) 之后, block 在 3.
    实际场景: iteration=10/10 + gate block → stop (预算优先).
    验证 block 不会覆盖预算耗尽."""
    ctx = DecisionContext(
        iteration=10, max_iterations=10,
        gate_status="block", bandit_advice="continue",
    )
    d = arbiter.evaluate(ctx)
    assert d.action == "stop"
    assert d.source == "budget_exhausted"
