"""实跑验证: 4 flag (A/B/C/D) 在真实 run_cognitive 路径上的行为.

selfcheck 只验 gating 分支选择 (用 __new__ 绕过 __init__), 本文件验完整
__init__ + 真实 run_cognitive + flag on, 看真实运行时会不会因状态字段缺失
/ 事件序列错乱 / 异步竞争而崩.

5 个 case:
  1. D flag 单开   — BranchIncubator N 路隔离采样
  2. B flag 单开   — ThreeCabin 三舱反射
  3. C flag 单开   — CompletionGate 三审放行
  4. A flag 单开   — CrossDomain 跨域迁移
  5. 4 flag 全开   — 叠加不冲突

复用 tests/test_autoloop_e2e.py 的 FakeLLM + _stub_heavy_calls 模式.
D flag 额外: 注入 agent_factory + 提前注入带 mock dispatch 的 BranchIncubator.

ponytail: 不调真 LLM (FakeLLM callable 按关键词路由), 不真起 Subagent
(_MockSubagentDispatch 返固定假说). 只验 engine 接入路径不崩 + phase 完成.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from huginn.autoloop.engine import AutoloopEngine, AutoloopResult
from huginn.autoloop.hypothesis_loop import HypothesisGraph
from huginn.autoloop.phase_gate import get_shared_phase_gate_state
from huginn.memory.manager import MemoryManager
from huginn.metacog.branch_incubator import BranchIncubator, _MockSubagentDispatch
from tests.fixtures.fake_llm import make_callable_llm
from tests.test_autoloop_e2e import (
    _DummyTracker,
    _StubBenchRunner,
    _bypass_validate_gate,
    _make_stage_llm,
    _restore_gate,
    _stub_heavy_calls,
)


def _make_engine_with_flags(
    tmp_path, fake_llm, monkeypatch, *, use_branch_incubator=False
):
    """构造 engine + 开 flag. D flag 时额外注入 agent_factory + mock dispatch.

    其他 flag (B/C/A) 直接 setenv 即可, engine 运行时读 env var.
    """
    _stub_heavy_calls(monkeypatch, fake_llm)
    memory = MemoryManager()

    agent_factory = object() if use_branch_incubator else None
    engine = AutoloopEngine(
        workspace=tmp_path,
        verification_model=fake_llm,
        memory_manager=memory,
        agent_factory=agent_factory,
    )
    engine.progress_tracker = _DummyTracker()
    engine._use_llm_decider = False
    engine._perceive = lambda: {
        "changed_files": ["diffusion_analysis.py"],
        "git_diff": "+def calc_diffusion(ca_si_ratio): ...",
        "timestamp": "2026-07-04T10:00:00Z",
        "goal": "Optimize C-S-H defect kinetics",
    }

    # D flag: 提前注入带 mock dispatch 的 BranchIncubator, 避免真起 Subagent
    if use_branch_incubator:
        engine._branch_incubator = BranchIncubator(
            dispatch=_MockSubagentDispatch(
                summary_template=(
                    "SELECTED: If Ca/Si > 1.2, interlayer spacing collapses "
                    "non-linearly via percolation threshold. [{family}]"
                )
            )
        )

    return engine, memory


def _drop_passing_test(tmp_path):
    (tmp_path / "test_smoke.py").write_text(
        "def test_ok():\n    assert True\n", encoding="utf-8"
    )


def _assert_run_completed(result: AutoloopResult, label: str):
    """通用断言: 不崩 + phase 全 completed + 关键阶段产出."""
    assert isinstance(result, AutoloopResult), f"[{label}] 不是 AutoloopResult"
    assert result.run_id.startswith("loop_"), f"[{label}] run_id 异常: {result.run_id}"
    assert result.objective == "Optimize C-S-H defect kinetics", f"[{label}] objective 异常"

    names = [p.name for p in result.phases]
    for expected in ("hypothesize", "plan", "execute", "validate", "learn", "report"):
        assert expected in names, f"[{label}] 缺 phase: {expected}, 实际 {names}"

    for phase in result.phases:
        assert phase.status == "completed", (
            f"[{label}] phase '{phase.name}' status={phase.status} error={phase.error}"
        )


# ── Case 1: D flag — BranchIncubator ─────────────────────────

@pytest.mark.asyncio
async def test_flag_D_branch_incubator(tmp_path, monkeypatch):
    """D flag on: _hypothesize 走 BranchIncubator N 路隔离采样路径."""
    monkeypatch.setenv("HUGINN_USE_BRANCH_INCUBATOR", "1")
    fake_llm = _make_stage_llm()
    engine, _ = _make_engine_with_flags(
        tmp_path, fake_llm, monkeypatch, use_branch_incubator=True
    )
    _drop_passing_test(tmp_path)

    gate_state = _bypass_validate_gate()
    try:
        result = await engine.run_cognitive(
            objective="Optimize C-S-H defect kinetics",
            max_iterations=5,
            progressive_budget=False,
        )
    finally:
        _restore_gate(gate_state)

    _assert_run_completed(result, "D-flag")

    # BranchIncubator 路径验证: mock dispatch 被调过
    mock_dispatch = engine._branch_incubator._dispatch
    assert len(mock_dispatch.calls) > 0, (
        "[D-flag] BranchIncubator dispatch 未被调用, 可能 fallback 到 2 路路径"
    )

    # 假设阶段产出了非空 hypothesis
    hyp_phase = next(p for p in result.phases if p.name == "hypothesize")
    assert hyp_phase.result, "[D-flag] hypothesize 返回空"


# ── Case 2: B flag — ThreeCabin ──────────────────────────────

@pytest.mark.asyncio
async def test_flag_B_three_cabin(tmp_path, monkeypatch):
    """B flag on: reflect_fn 走 ThreeCabin 三舱反射 (真 StepEvaluation)."""
    monkeypatch.setenv("HUGINN_USE_THREE_CABIN", "1")
    fake_llm = _make_stage_llm()
    engine, _ = _make_engine_with_flags(tmp_path, fake_llm, monkeypatch)
    _drop_passing_test(tmp_path)

    gate_state = _bypass_validate_gate()
    try:
        result = await engine.run_cognitive(
            objective="Optimize C-S-H defect kinetics",
            max_iterations=5,
            progressive_budget=False,
        )
    finally:
        _restore_gate(gate_state)

    _assert_run_completed(result, "B-flag")


# ── Case 3: C flag — CompletionGate ──────────────────────────

@pytest.mark.asyncio
async def test_flag_C_completion_gate(tmp_path, monkeypatch):
    """C flag on: reflect_fn 走 CompletionGate 三审统一入口."""
    monkeypatch.setenv("HUGINN_USE_COMPLETION_GATE", "1")
    fake_llm = _make_stage_llm()
    engine, _ = _make_engine_with_flags(tmp_path, fake_llm, monkeypatch)
    _drop_passing_test(tmp_path)

    gate_state = _bypass_validate_gate()
    try:
        result = await engine.run_cognitive(
            objective="Optimize C-S-H defect kinetics",
            max_iterations=5,
            progressive_budget=False,
        )
    finally:
        _restore_gate(gate_state)

    _assert_run_completed(result, "C-flag")


# ── Case 4: A flag — CrossDomain ─────────────────────────────

@pytest.mark.asyncio
async def test_flag_A_cross_domain(tmp_path, monkeypatch):
    """A flag on: _conjecture_hint 优先调 cross_domain_reframe 跨域迁移."""
    monkeypatch.setenv("HUGINN_USE_CROSS_DOMAIN", "1")
    fake_llm = _make_stage_llm()
    engine, _ = _make_engine_with_flags(tmp_path, fake_llm, monkeypatch)
    _drop_passing_test(tmp_path)

    gate_state = _bypass_validate_gate()
    try:
        result = await engine.run_cognitive(
            objective="Optimize C-S-H defect kinetics",
            max_iterations=5,
            progressive_budget=False,
        )
    finally:
        _restore_gate(gate_state)

    _assert_run_completed(result, "A-flag")


# ── Case 5: 4 flag 全开 ──────────────────────────────────────

@pytest.mark.asyncio
async def test_all_4_flags_combined(tmp_path, monkeypatch):
    """4 flag 同时开: 验证叠加不冲突, 各路径正常触发."""
    monkeypatch.setenv("HUGINN_USE_BRANCH_INCUBATOR", "1")
    monkeypatch.setenv("HUGINN_USE_THREE_CABIN", "1")
    monkeypatch.setenv("HUGINN_USE_COMPLETION_GATE", "1")
    monkeypatch.setenv("HUGINN_USE_CROSS_DOMAIN", "1")

    fake_llm = _make_stage_llm()
    engine, _ = _make_engine_with_flags(
        tmp_path, fake_llm, monkeypatch, use_branch_incubator=True
    )
    _drop_passing_test(tmp_path)

    gate_state = _bypass_validate_gate()
    try:
        result = await engine.run_cognitive(
            objective="Optimize C-S-H defect kinetics",
            max_iterations=5,
            progressive_budget=False,
        )
    finally:
        _restore_gate(gate_state)

    _assert_run_completed(result, "4-flags-combined")

    # D flag 路径验证
    mock_dispatch = engine._branch_incubator._dispatch
    assert len(mock_dispatch.calls) > 0, (
        "[4-flags] BranchIncubator dispatch 未被调用"
    )
