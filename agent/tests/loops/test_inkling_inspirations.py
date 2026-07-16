"""Self-check tests for the three Inkling-inspired changes.

1. RSI directive: learn 阶段写 _next_loop_directive, 下一轮注入 speculator_hint
2. Controllable thinking effort: phase→effort→prompt 映射
3. Tool order randomization: 打乱后元素不变, 顺序变

按 ponytail: 最小检查, 不建框架, 不做 fixture. 只验证逻辑核心.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest


# ── 1. Thinking effort mapping ─────────────────────────────────


def test_effort_to_prompt_thresholds():
    """_effort_to_prompt 把 0-1 连续值映射到 3 档 prompt 指令."""
    from huginn.autoloop.engine import _effort_to_prompt, _PHASE_THINKING_EFFORT

    # 高 effort (>= 0.8) → 深度推理指令
    high = _effort_to_prompt(0.9)
    assert "deeply" in high or "step-by-step" in high, (
        f"high effort should ask for deep thinking, got: {high}"
    )

    # 中 effort (0.5-0.7) → 中等指令
    mid = _effort_to_prompt(0.6)
    assert "concisely" in mid or "carefully" in mid, (
        f"mid effort should ask for concise reasoning, got: {mid}"
    )

    # 低 effort (< 0.5) → 直接回答
    low = _effort_to_prompt(0.3)
    assert "directly" in low or "briefly" in low, (
        f"low effort should ask for direct answer, got: {low}"
    )

    # 所有 phase 的 effort 值都在 [0, 1] 范围内
    for phase, effort in _PHASE_THINKING_EFFORT.items():
        assert 0.0 <= effort <= 1.0, (
            f"phase '{phase}' effort {effort} out of [0,1] range"
        )


def test_phase_thinking_effort_covers_all_phases():
    """每个 autoloop phase 都有对应的 thinking effort 配置."""
    from huginn.autoloop.engine import AUTOLOOP_PHASES, _PHASE_THINKING_EFFORT

    for phase in AUTOLOOP_PHASES:
        assert phase in _PHASE_THINKING_EFFORT, (
            f"phase '{phase}' missing from _PHASE_THINKING_EFFORT"
        )


# ── 2. RSI directive injection (memory-backed) ──────────────────


def test_rsi_uses_memory_not_prompt_field():
    """RSI directive 不应该再用 _next_loop_directive 字段, 应该走 memory."""
    from huginn.autoloop.engine import AutoloopEngine
    import inspect

    # __init__ 不应该有 _next_loop_directive 字段 (已迁到 memory)
    src_init = inspect.getsource(AutoloopEngine.__init__)
    assert "_next_loop_directive" not in src_init, (
        "_next_loop_directive field should be removed — directive now goes to memory"
    )

    # _generate_next_loop_directive 应该调 memory.remember
    src_gen = inspect.getsource(AutoloopEngine._generate_next_loop_directive)
    assert "memory.remember" in src_gen or "self.memory.remember" in src_gen, (
        "_generate_next_loop_directive should write to memory.remember"
    )


def test_rsi_no_prompt_injection_in_main_loop():
    """主循环 run() 不应该有 directive 注入 speculator_hint 的逻辑."""
    from huginn.autoloop.engine import AutoloopEngine
    import inspect

    src_run = inspect.getsource(AutoloopEngine.run)
    assert "_next_loop_directive" not in src_run, (
        "run() should not reference _next_loop_directive — directive flows via memory"
    )


@pytest.mark.asyncio
async def test_generate_next_loop_directive_writes_memory():
    """_generate_next_loop_directive 调 LLM 后把 directive 写入 memory.remember."""
    from huginn.autoloop.engine import AutoloopEngine

    engine = AutoloopEngine.__new__(AutoloopEngine)
    engine._iteration = 3
    engine._llm_chat = AsyncMock(return_value="Avoid RBF kernel, try Tanimoto next time.")

    # mock memory.remember 捕获调用参数
    remembered = []
    engine.memory = MagicMock()
    engine.memory.remember = lambda **kw: remembered.append(kw)

    await engine._generate_next_loop_directive(
        hypothesis="GP with RBF kernel will work",
        plan={"mode": "coder"},
        validation={"tests_passed": False, "prediction_error": {"surprise": 0.8}},
        r_phys=0.2,
    )

    assert len(remembered) == 1, "memory.remember should be called exactly once"
    entry = remembered[0]
    assert entry["category"] == "self_directive"
    assert "rsi" in entry["tags"]
    assert "Tanimoto" in entry["content"]
    assert entry["tier"] == "mid"
    # importance 跟 surprise 挂钩: surprise=0.8 → importance ≈ 0.5 + 0.32 = 0.82
    assert entry["importance"] > 0.7, (
        f"high surprise should boost importance, got {entry['importance']}"
    )


@pytest.mark.asyncio
async def test_generate_next_loop_directive_fails_silently():
    """LLM call 失败时方法自身捕获异常, 不写 memory, 不抛."""
    from huginn.autoloop.engine import AutoloopEngine

    engine = AutoloopEngine.__new__(AutoloopEngine)
    engine._iteration = 1
    engine._llm_chat = AsyncMock(side_effect=RuntimeError("API down"))
    engine.memory = MagicMock()

    # 不应该抛
    try:
        await engine._generate_next_loop_directive(
            hypothesis="test",
            plan={"mode": "coder"},
            validation={},
            r_phys=None,
        )
    except RuntimeError:
        pytest.fail("_generate_next_loop_directive should catch LLM errors internally")

    # LLM 挂了, memory.remember 不应该被调
    engine.memory.remember.assert_not_called()


# ── 3. Tool order randomization ─────────────────────────────────


def test_randomize_tool_order_preserves_elements():
    """打乱后工具数量和元素不变, 只是顺序变."""
    from huginn.bench.tool_randomization import randomize_tool_order

    tools = [f"tool_{i}" for i in range(10)]
    shuffled = randomize_tool_order(tools, seed=42)

    assert len(shuffled) == len(tools)
    assert set(shuffled) == set(tools), "elements changed after shuffle"


def test_randomize_tool_order_is_deterministic_with_seed():
    """同 seed 产生同顺序, 保证 benchmark 可复现."""
    from huginn.bench.tool_randomization import randomize_tool_order

    tools = [f"tool_{i}" for i in range(10)]
    s1 = randomize_tool_order(tools, seed=123)
    s2 = randomize_tool_order(tools, seed=123)

    assert s1 == s2, "same seed should produce same order"


def test_randomize_tool_order_changes_order():
    """打乱后顺序确实变了 (统计意义上)."""
    from huginn.bench.tool_randomization import randomize_tool_order

    tools = [f"tool_{i}" for i in range(20)]
    shuffled = randomize_tool_order(tools, seed=999)

    # 20 个元素, 完全不变的概率极低
    differences = sum(1 for a, b in zip(tools, shuffled) if a != b)
    assert differences > 0, "shuffle didn't change order at all"


# ── Smoke: 跑一遍确认 import 不挂 ────────────────────────────────


def test_all_imports_ok():
    """三个改动的 import 都能成功."""
    from huginn.autoloop.engine import (
        _effort_to_prompt,
        _PHASE_THINKING_EFFORT,
        AutoloopEngine,
    )
    from huginn.bench.tool_randomization import randomize_tool_order

    assert callable(_effort_to_prompt)
    assert isinstance(_PHASE_THINKING_EFFORT, dict)
    assert callable(randomize_tool_order)
