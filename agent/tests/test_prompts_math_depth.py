"""E1: prompts.py 数学深度引导同步 — 验证 HUGINN_SYSTEM_PROMPT 含数学深度块.

覆盖:
- 关键 action 名 (pde_classify / euler_lagrange / noether / diffgeo_metric /
  diffgeo_curvature / sobol_indices / constraint_check) 出现在 system prompt
- 默认 persona 继承该块
- 与 engine.py 的 _MATH_DEPTH_PROMPT_BLOCK 在工具清单上一致
- 决策规则 (PDE/变分/守恒律优先) 在 prompt 里
"""

from __future__ import annotations

import re

from huginn.prompts import HUGINN_SYSTEM_PROMPT
from huginn.personas import Persona, PersonaManager


REQUIRED_ACTIONS = [
    "pde_classify",
    "pde_separation",
    "pde_characteristics",
    "pde_discretize",
    "euler_lagrange",
    "derive",
    "functional_derivative",
    "isoperimetric",
    "noether",
    "diffgeo_metric",
    "diffgeo_geodesic",
    "diffgeo_curvature",
    "diffgeo_lie_derivative",
    "diffgeo_connection",
    "sobol_indices",
    "constraint_check",
]


class TestPromptsMathDepthBlock:
    def test_math_depth_section_header_present(self):
        assert "## Math Depth Guidance" in HUGINN_SYSTEM_PROMPT

    def test_physics_is_math_stance_present(self):
        # 用户核心理念: 物理化学本质上是数学
        assert "treat physics/chemistry as mathematics" in HUGINN_SYSTEM_PROMPT

    def test_all_math_actions_listed(self):
        missing = [a for a in REQUIRED_ACTIONS if a not in HUGINN_SYSTEM_PROMPT]
        assert missing == [], f"prompt 缺少 action: {missing}"

    def test_decision_rule_present(self):
        # 决策规则: PDE/变分/守恒律 优先于数值求解
        assert "PDE" in HUGINN_SYSTEM_PROMPT
        assert "variational" in HUGINN_SYSTEM_PROMPT.lower()
        assert "conservation law" in HUGINN_SYSTEM_PROMPT.lower()
        assert "symbolically" in HUGINN_SYSTEM_PROMPT.lower()

    def test_phase_gate_forecast_present(self):
        # 提前埋钩子: validate→learn 门会检查数学证据
        assert "phase-gate" in HUGINN_SYSTEM_PROMPT.lower()
        assert "validate" in HUGINN_SYSTEM_PROMPT.lower()


class TestDefaultPersonaInheritsMathBlock:
    def test_default_persona_uses_huginn_system_prompt(self, tmp_path):
        mgr = PersonaManager(workspace=tmp_path)
        default = mgr.get("default")
        assert isinstance(default, Persona)
        # default persona 的 system_prompt 应当包含数学深度块
        assert "## Math Depth Guidance" in default.system_prompt
        assert "pde_classify" in default.system_prompt

    def test_dft_persona_sees_math_block(self, tmp_path):
        # 即使是 DFT 专家也应当看到数学深度引导 — 物理即数学
        mgr = PersonaManager(workspace=tmp_path)
        dft = mgr.get("dft_expert")
        if dft and dft.system_prompt:
            # dft_expert 可能拼接 HUGINN_SYSTEM_PROMPT 或独立定义
            # 至少应当提到 PDE 或变分原理
            prompt_lower = dft.system_prompt.lower()
            assert ("pde" in prompt_lower) or ("variational" in prompt_lower) or (
                "math" in prompt_lower
            )


class TestEnginePromptConsistency:
    def test_engine_math_block_actions_subset_of_prompts(self):
        # engine.py 的 _MATH_DEPTH_PROMPT_BLOCK 提到的 action 应当都在 system prompt 里
        # 保证 persona 注入和 prompt 注入说的是同一套工具
        from huginn.autoloop.engine import AutoloopEngine

        engine_block = AutoloopEngine._MATH_DEPTH_PROMPT_BLOCK
        for action in ["pde_classify", "euler_lagrange", "noether",
                       "diffgeo_metric", "diffgeo_curvature",
                       "sobol_indices", "constraint_check"]:
            assert action in engine_block, f"engine block 缺 {action}"
            assert action in HUGINN_SYSTEM_PROMPT, (
                f"system prompt 缺 {action} (engine 有) — 两边不同步"
            )

    def test_engine_block_mentions_symbolic_first(self):
        from huginn.autoloop.engine import AutoloopEngine

        engine_block = AutoloopEngine._MATH_DEPTH_PROMPT_BLOCK.lower()
        # engine 强调符号优先
        assert "symbolic" in engine_block
