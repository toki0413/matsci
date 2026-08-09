"""Tests for math-depth prompt injection in AutoloopEngine.

Verifies that _build_hypothesis_prompt and _build_plan_prompt both
include the _MATH_DEPTH_PROMPT_BLOCK that nudges the agent toward
PDE / variational / diffgeo / symreg reasoning.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from huginn.autoloop.engine import AutoloopEngine


@pytest.fixture
def engine() -> AutoloopEngine:
    """Build a minimal AutoloopEngine without invoking __init__
    (which would call get_settings/get_model and pull in real config).
    We only need the prompt-building methods, which depend on:
      - self._speculator_hint (None)
      - self._build_kb_text (returns "")
      - self.workspace (str)
    """
    eng = AutoloopEngine.__new__(AutoloopEngine)
    eng._speculator_hint = None
    eng._kb = None
    eng.workspace = "."
    # E2-1: MCMC 状态字段 (真实 __init__ 里初始化, __new__ 下手动补齐)
    eng._iteration = 0
    eng._hypo_manifold = None
    eng._mcmc_current = None
    eng._mcmc_rng = None
    eng._mcmc_rng_state = None
    eng._mcmc_cached_log_p = None
    eng._mcmc_step_count = 0
    eng._mcmc_accept_count = 0
    eng._mcmc_chains = {}
    # _build_kb_text 在 KB 未初始化时应返回空串 — 但若实现依赖 self._kb,
    # 我们直接 monkeypatch 一个返回空串的版本以隔离 ChromaDB.
    eng._build_kb_text = lambda query: ""  # type: ignore[method-assign]
    return eng


class TestMathPromptInjection:
    """Math-depth block must appear in hypothesis + plan prompts."""

    def test_hypothesis_prompt_contains_math_block(self, engine):
        prompt = engine._build_hypothesis_prompt(
            context={"objective": "find band gap of GaN"}
        )
        # 关键词检查: 数学深度引导块应该出现
        assert "Math depth guidance" in prompt
        assert "pde_classify" in prompt
        assert "euler_lagrange" in prompt
        assert "diffgeo_metric" in prompt
        assert "sobol_indices" in prompt
        assert "constraint_check" in prompt

    def test_plan_prompt_contains_math_block(self, engine):
        prompt = engine._build_plan_prompt(
            hypothesis="derive heat equation for thermal transport",
            context={"objective": "thermal"},
        )
        assert "Math depth guidance" in prompt
        assert "pde_classify" in prompt
        assert "noether" in prompt

    def test_math_block_is_class_constant(self):
        # _MATH_DEPTH_PROMPT_BLOCK 是类常量, 不依赖实例状态
        assert hasattr(AutoloopEngine, "_MATH_DEPTH_PROMPT_BLOCK")
        block = AutoloopEngine._MATH_DEPTH_PROMPT_BLOCK
        assert isinstance(block, str)
        assert "physics/chemistry" in block

    def test_hypothesis_prompt_mentions_pde_preference(self, engine):
        # 引导文本应明确建议先识别 PDE 结构再上数值实验
        prompt = engine._build_hypothesis_prompt(context={})
        assert "Prefer hypotheses that can be expressed as governing PDEs" in prompt

    def test_plan_prompt_mentions_symbolic_first(self, engine):
        # planner should mention symbolic_math_tool actions alongside numerical solvers
        prompt = engine._build_plan_prompt(hypothesis="test", context={})
        assert "symbolic_math_tool actions" in prompt
        assert "numerical solvers" in prompt


class TestE2_1McmcAdvance:
    """E2-1: 通用主循环必须推进 MCMC 链 (原本只在 rcb_runner 跑).

    _build_hypothesis_prompt 每 HUGINN_MCMC_INTERVAL 轮推进一次 mcmc_step,
    把采样链驻留点注入 hint, 补充 abductive_inference 的贪婪盲区.
    """

    def test_mcmc_current_advances(self, engine):
        # 含白名单指标 (accuracy/mae) 的 context → 能抽取观测 → 触发 MCMC.
        ctx = {"goal": "reproduce accuracy=0.85 mae=0.42 from paper"}
        for i in range(1, 6):  # interval=5, 到第 5 轮应推进 1 步
            engine._iteration = i
            engine._build_hypothesis_prompt(ctx)
        assert engine._mcmc_step_count >= 1, "MCMC step 未推进"
        assert engine._mcmc_current is not None, "MCMC current 仍为 None"

    def test_mcmc_hint_injected(self, engine):
        ctx = {"goal": "reproduce accuracy=0.85 mae=0.42 from paper"}
        for i in range(1, 6):
            engine._iteration = i
            prompt = engine._build_hypothesis_prompt(ctx)
        # 推进后 hint 应包含 MCMC 驻留假设段
        assert "[posterior mcmc hint]" in prompt

    def test_no_obs_no_mcmc(self, engine):
        # 无可抽取观测 → 不推进, 不报错 (降级路径)
        engine._iteration = 5
        engine._build_hypothesis_prompt({"goal": "no numeric target here"})
        assert engine._mcmc_step_count == 0
