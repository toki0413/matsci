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
