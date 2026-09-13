"""Self-check tests for the three Inkling-inspired changes.

1. RSI directive: learn 阶段写 _next_loop_directive, 下一轮注入 speculator_hint
2. Controllable thinking effort: phase→effort→prompt 映射
3. Tool order randomization: 打乱后元素不变, 顺序变

按 ponytail: 最小检查, 不建框架, 不做 fixture. 只验证逻辑核心.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# ── 1. Thinking effort mapping ─────────────────────────────────


def test_effort_to_prompt_thresholds():
    """_effort_to_prompt 把 0-1 连续值映射到 3 档 prompt 指令."""
    from huginn.autoloop.engine import _PHASE_THINKING_EFFORT, _effort_to_prompt

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
    from huginn.autoloop.engine import _PHASE_THINKING_EFFORT, AUTOLOOP_PHASES

    for phase in AUTOLOOP_PHASES:
        assert phase in _PHASE_THINKING_EFFORT, (
            f"phase '{phase}' missing from _PHASE_THINKING_EFFORT"
        )


# ── 2. RSI directive injection (memory-backed) ──────────────────


def test_rsi_uses_memory_not_prompt_field():
    """RSI directive 不应该再用 _next_loop_directive 字段, 应该走 memory."""
    import inspect

    from huginn.autoloop.engine import AutoloopEngine
    from huginn.autoloop.engine_reflect import EngineReflect

    # __init__ 不应该有 _next_loop_directive 字段 (已迁到 memory)
    src_init = inspect.getsource(AutoloopEngine.__init__)
    assert "_next_loop_directive" not in src_init, (
        "_next_loop_directive field should be removed — directive now goes to memory"
    )

    # _generate_next_loop_directive 应该调 memory.remember.
    # 去 mixin 阶段8: 该方法已下沉为 EngineReflect 协作对象, 引擎上只剩薄委托,
    # 所以源码断言要指向真实实现 (EngineReflect), 而不是引擎委托.
    src_gen = inspect.getsource(EngineReflect._generate_next_loop_directive)
    assert "memory.remember" in src_gen or "self.memory.remember" in src_gen, (
        "_generate_next_loop_directive should write to memory.remember"
    )


def test_rsi_no_prompt_injection_in_main_loop():
    """主循环 run() 不应该有 directive 注入 speculator_hint 的逻辑."""
    import inspect

    from huginn.autoloop.engine import AutoloopEngine

    src_run = inspect.getsource(AutoloopEngine.run_cognitive)
    assert "_next_loop_directive" not in src_run, (
        "run_cognitive() should not reference _next_loop_directive — directive flows via memory"
    )


def _build_directive_engine(directive_fn):
    """真实依赖构造: real EngineSignals(SignalBridge) + real MemoryManager + 真实
    _llm_chat 适配器 (委托给一个真正可调用 directive_fn, 非 unittest.mock).

    去 mixin 阶段8: _generate_next_loop_directive 已下沉为 EngineReflect 协作对象,
    引擎 __new__ 绕过 __init__ 时需手动挂 real EngineSignals (SignalBridge 前置)."""
    import tempfile

    from huginn.autoloop.engine import AutoloopEngine
    from huginn.autoloop.engine_reflect import EngineReflect
    from huginn.autoloop.signals import EngineSignals
    from huginn.memory.manager import MemoryManager

    engine = AutoloopEngine.__new__(AutoloopEngine)
    # 去 mixin 阶段8: _generate_next_loop_directive 是 EngineReflect 协作对象方法,
    # 引擎对象 __new__ 绕过 __init__ 需手动挂 _engine_reflector.
    engine._engine_reflector = EngineReflect(engine)
    engine.signals = EngineSignals()
    engine._iteration = 3
    # 真实 _llm_chat 适配器: 委托给一个真正可调用 directive_fn (非 unittest.mock).
    # EngineReflect._generate_next_loop_directive 经属性转发读到这个真实函数.
    async def _llm_chat(prompt: str, **kw):
        return directive_fn(prompt)

    engine._llm_chat = _llm_chat
    # 真实 long-term memory, 落临时 sqlite, 不污染 ~/.huginn.
    tmpdir = Path(tempfile.mkdtemp(prefix="inkling-mem-"))
    from huginn.memory.manager import MemoryConfig

    engine.memory = MemoryManager(config=MemoryConfig(memory_dir=tmpdir))
    return engine


def _directive_memory_entries(engine):
    """从真实 MemoryManager 读回 self_directive 类记忆."""
    return engine.memory.recall("self_directive", category="self_directive", top_k=20)


@pytest.mark.asyncio
async def test_generate_next_loop_directive_writes_memory():
    """_generate_next_loop_directive 调真实 LLM 后把 directive 写入真实 memory."""
    from huginn.autoloop.engine import AutoloopEngine

    engine = _build_directive_engine(
        lambda prompt: "Avoid RBF kernel, try Tanimoto next time."
    )

    await engine._generate_next_loop_directive(
        hypothesis="GP with RBF kernel will work",
        plan={"mode": "coder"},
        validation={"tests_passed": False, "prediction_error": {"surprise": 0.8}},
        r_phys=0.2,
    )

    entries = _directive_memory_entries(engine)
    assert len(entries) >= 1, "self_directive should be stored in real memory"
    top = entries[0]
    assert top.get("category") == "self_directive"
    tags = top.get("tags") or []
    assert "rsi" in tags
    assert "Tanimoto" in top.get("content", "")
    assert top.get("tier") == "mid"
    # importance 跟 surprise 挂钩: surprise=0.8 → importance ≈ 0.5 + 0.32 = 0.82
    assert top.get("importance", 0) > 0.7, (
        f"high surprise should boost importance, got {top.get('importance')}"
    )


@pytest.mark.asyncio
async def test_generate_next_loop_directive_fails_silently():
    """真实 LLM 调用抛异常时方法自身捕获, 不写 memory, 不抛."""
    def _boom(prompt: str):
        raise RuntimeError("API down")

    engine = _build_directive_engine(_boom)

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

    # LLM 挂了, 真实 memory 里不该出现 self_directive
    entries = _directive_memory_entries(engine)
    assert not any(
        "self-directive" in (e.get("content") or "") for e in entries
    ), "LLM 失败时不应写 self_directive 记忆"


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
        _PHASE_THINKING_EFFORT,
        _effort_to_prompt,
    )
    from huginn.bench.tool_randomization import randomize_tool_order

    assert callable(_effort_to_prompt)
    assert isinstance(_PHASE_THINKING_EFFORT, dict)
    assert callable(randomize_tool_order)
