"""Loop engineering 审计回归测试.

验证:
1. 所有 _KNOWN_TOOLS 都有至少一条 PIPELINE_RULE (无孤立工具)
2. 所有 next_tool_hints 里的工具都有自己的 rule (无悬空引用)
3. 所有 _KNOWN_TOOLS 都有 _infer_stage 映射 (无未分类工具)
4. 新增 hook 对正确的 tool_name 响应
"""

from __future__ import annotations

import asyncio

import pytest

from huginn.hooks import HookContext
from huginn.provenance.pipeline import (
    PIPELINE_RULES,
    PipelineStage,
    _KNOWN_TOOLS,
    _infer_stage,
)


# ── 1. 孤立工具检查 ──────────────────────────────────────────────


def _all_trigger_tools() -> set[str]:
    """PIPELINE_RULES 里所有作为触发工具出现过的 tool_name."""
    return {r.tool_name for r in PIPELINE_RULES}


def _all_hinted_tools() -> set[str]:
    """PIPELINE_RULES 里所有在 next_tool_hints 中被建议过的工具."""
    hints: set[str] = set()
    for r in PIPELINE_RULES:
        hints.update(r.next_tool_hints)
    return hints


class TestNoOrphanedTools:
    """所有管线认识的工具都应该有至少一条规则."""

    # 这些是管线终端, 只做分析不做后续建议
    _TERMINAL_TOOLS = {"compute_msd", "compute_rdf"}

    def test_known_tools_have_rules(self):
        triggers = _all_trigger_tools()
        orphans = [t for t in _KNOWN_TOOLS if t not in triggers and t not in self._TERMINAL_TOOLS]
        assert not orphans, f"孤立工具 (有 _KNOWN_TOOLS 但无 PIPELINE_RULE): {orphans}"

    def test_known_tools_have_stage_inference(self):
        no_stage = []
        for tool in _KNOWN_TOOLS:
            stage = _infer_stage(tool, {"action": "relax"})
            if stage is None:
                stage = _infer_stage(tool, {})
            if stage is None:
                no_stage.append(tool)
        assert not no_stage, f"无法推断阶段的工具: {no_stage}"


# ── 2. 悬空引用检查 ──────────────────────────────────────────────


class TestNoDanglingReferences:
    """所有在 hints 中被建议的工具应该有自己的 rule (除非是终端工具)."""

    # 这些是管线终端, 被建议但不触发后续规则是正常的
    _TERMINAL_TOOLS = {"compute_msd", "compute_rdf"}

    def test_hinted_tools_have_own_rules(self):
        triggers = _all_trigger_tools()
        hinted = _all_hinted_tools()
        dangling = hinted - triggers - self._TERMINAL_TOOLS
        assert not dangling, (
            f"悬空引用 (被 hint 但自己没有 rule): {dangling}"
        )


# ── 3. 互补断链验证 ──────────────────────────────────────────────


class TestComplementaryChains:
    """验证关键互补桥接规则存在."""

    def _find_rules(self, tool_name: str) -> list:
        return [r for r in PIPELINE_RULES if r.tool_name == tool_name]

    def test_neb_to_enhanced_sampling(self):
        rules = self._find_rules("neb_tool")
        hints = set()
        for r in rules:
            hints.update(r.next_tool_hints)
        assert "enhanced_sampling_tool" in hints, "NEB → enhanced_sampling 断链"

    def test_tda_to_msm(self):
        rules = self._find_rules("tda_tool")
        hints = set()
        for r in rules:
            hints.update(r.next_tool_hints)
        assert "msm_tool" in hints, "TDA → MSM 断链"

    def test_descriptor_to_inverse_design(self):
        rules = self._find_rules("descriptor_tool")
        hints = set()
        for r in rules:
            hints.update(r.next_tool_hints)
        assert "inverse_design_tool" in hints, "descriptor → inverse_design 断链"

    def test_gp_to_active_learning(self):
        rules = self._find_rules("gp_tool")
        hints = set()
        for r in rules:
            hints.update(r.next_tool_hints)
        assert "active_learning_tool" in hints, "GP → active_learning 断链"

    def test_multi_fidelity_to_consensus(self):
        rules = self._find_rules("multi_fidelity_tool")
        hints = set()
        for r in rules:
            hints.update(r.next_tool_hints)
        assert "consensus_scoring_tool" in hints, "multi_fidelity → consensus 断链"

    def test_thermo_to_fep(self):
        rules = self._find_rules("thermo_tool")
        hints = set()
        for r in rules:
            hints.update(r.next_tool_hints)
        assert "fep_tool" in hints, "thermo → FEP 断链"

    def test_symmetry_to_motif_mining(self):
        rules = self._find_rules("symmetry_tool")
        hints = set()
        for r in rules:
            hints.update(r.next_tool_hints)
        assert "motif_mining_tool" in hints, "symmetry → motif_mining 断链"

    def test_ml_potential_to_lammps(self):
        rules = self._find_rules("ml_potential_tool")
        hints = set()
        for r in rules:
            hints.update(r.next_tool_hints)
        assert "lammps_tool" in hints, "ml_potential → lammps 断链"

    def test_generative_design_to_inverse_design(self):
        rules = self._find_rules("generative_design_tool")
        hints = set()
        for r in rules:
            hints.update(r.next_tool_hints)
        assert "inverse_design_tool" in hints, "generative_design → inverse_design 断链"

    def test_characterization_has_exit(self):
        rules = self._find_rules("characterization_tool")
        assert len(rules) > 0, "characterization_tool 没有出口规则 (悬空引用)"

    def test_gaussian_has_rule(self):
        rules = self._find_rules("gaussian_tool")
        assert len(rules) > 0, "gaussian_tool 没有管线规则 (孤立入口)"

    def test_orca_has_rule(self):
        rules = self._find_rules("orca_tool")
        assert len(rules) > 0, "orca_tool 没有管线规则 (孤立入口)"

    def test_xrd_sim_has_rule(self):
        rules = self._find_rules("xrd_sim_tool")
        assert len(rules) > 0, "xrd_sim_tool 没有管线规则 (孤立入口)"

    def test_dynamics_discovery_has_rule(self):
        rules = self._find_rules("dynamics_discovery_tool")
        assert len(rules) > 0, "dynamics_discovery_tool 没有管线规则 (孤立入口)"

    def test_high_throughput_has_rule(self):
        rules = self._find_rules("high_throughput_tool")
        assert len(rules) > 0, "high_throughput_tool 没有管线规则 (孤立入口)"


# ── 4. Stage 推断完整性 ──────────────────────────────────────────


class TestStageInference:
    """验证新工具的阶段推断正确."""

    @pytest.mark.parametrize("tool,expected", [
        ("neb_tool", PipelineStage.PROPERTIES),
        ("tda_tool", PipelineStage.ANALYSIS),
        ("descriptor_tool", PipelineStage.PROPERTIES),
        ("gp_tool", PipelineStage.INVERSE_DESIGN),
        ("multi_fidelity_tool", PipelineStage.INVERSE_DESIGN),
        ("thermo_tool", PipelineStage.PROPERTIES),
        ("symmetry_tool", PipelineStage.PROPERTIES),
        ("ml_potential_tool", PipelineStage.STRUCTURE),
        ("generative_design_tool", PipelineStage.INVERSE_DESIGN),
        ("xrd_sim_tool", PipelineStage.PROPERTIES),
        ("dynamics_discovery_tool", PipelineStage.ANALYSIS),
        ("high_throughput_tool", PipelineStage.PROPERTIES),
        ("active_learning_tool", PipelineStage.INVERSE_DESIGN),
        ("uq_tool", PipelineStage.ANALYSIS),
        ("symbolic_regression_tool", PipelineStage.INVERSE_DESIGN),
        ("evidence_fusion_tool", PipelineStage.CONSENSUS),
        ("gaussian_tool", PipelineStage.RELAX),
        ("orca_tool", PipelineStage.RELAX),
    ])
    def test_stage_inference(self, tool, expected):
        stage = _infer_stage(tool, {})
        assert stage == expected, f"{tool} 阶段推断错误: 期望 {expected}, 得到 {stage}"

    def test_gaussian_static(self):
        assert _infer_stage("gaussian_tool", {"action": "scf"}) == PipelineStage.STATIC
        assert _infer_stage("gaussian_tool", {"action": "sp"}) == PipelineStage.STATIC

    def test_orca_static(self):
        assert _infer_stage("orca_tool", {"action": "scf"}) == PipelineStage.STATIC


# ── 5. Hook 响应验证 ─────────────────────────────────────────────


def _make_ctx(tool_name: str, result: dict) -> HookContext:
    """构造一个模拟的 HookContext."""
    return HookContext(
        tool_name=tool_name,
        args={},
        result=result,
        error=None,
        metadata={},
    )


class TestNewHooks:
    """验证新增 hook 对正确的工具响应且不误触其他工具."""

    @pytest.mark.parametrize("hook_func,tool_name,bad_result,should_block", [
        ("fep_validation_hook", "fep_tool", {"result": {"delta_g_kcal_mol": 999.0}}, False),
        ("fep_validation_hook", "fep_tool", {"result": {"text": "NaN detected"}}, True),
        ("enhanced_sampling_hook", "enhanced_sampling_tool", {"result": {"converged": False}}, False),
        ("enhanced_sampling_hook", "enhanced_sampling_tool", {"result": {"text": "inf"}}, True),
        ("msm_validation_hook", "msm_tool", {"result": {"transition_matrix": [[0.3, 0.3], [0.3, 0.3]]}}, False),
        ("msm_validation_hook", "msm_tool", {"result": {"text": "nan"}}, True),
        ("inverse_design_hook", "inverse_design_tool", {"result": {"pareto_indices": []}}, True),
        ("inverse_design_hook", "inverse_design_tool", {"result": {"pareto_indices": [0, 1]}}, False),
        ("consensus_scoring_hook", "consensus_scoring_tool", {"result": {"ranking": []}}, True),
        ("consensus_scoring_hook", "consensus_scoring_tool", {"result": {"ranking": [1, 2]}}, False),
        ("rdkit_validation_hook", "rdkit_tool", {"result": {"valid": False, "error": "bad smiles"}}, True),
        ("rdkit_validation_hook", "rdkit_tool", {"result": {"valid": True}}, False),
        ("neb_convergence_hook", "neb_tool", {"result": {"converged": False}}, False),
        ("neb_convergence_hook", "neb_tool", {"result": {"barrier_ev": -0.5}}, False),
        ("gp_model_hook", "gp_tool", {"result": {"r2_score": 0.1}}, False),
        ("gp_model_hook", "gp_tool", {"result": {"r2_score": 0.9}}, False),
    ])
    async def test_hook_response(self, hook_func, tool_name, bad_result, should_block):
        import huginn.hooks.science_hooks as mod

        hook = getattr(mod, hook_func)
        ctx = _make_ctx(tool_name, bad_result)
        await hook(ctx)

        if should_block:
            assert ctx.metadata.get("blocked_by_hook") is True, (
                f"{hook_func} 应该 block {tool_name} 但没有"
            )
        else:
            # 不该 block (可能 warn 或什么都不做)
            assert ctx.metadata.get("blocked_by_hook") is not True, (
                f"{hook_func} 不应该 block {tool_name} 但 block 了"
            )

    @pytest.mark.parametrize("hook_func", [
        "fep_validation_hook",
        "enhanced_sampling_hook",
        "msm_validation_hook",
        "inverse_design_hook",
        "motif_mining_hook",
        "consensus_scoring_hook",
        "rdkit_validation_hook",
        "neb_convergence_hook",
        "gp_model_hook",
    ])
    async def test_hook_ignores_wrong_tool(self, hook_func):
        """hook 不应该对其他工具的输出做任何操作."""
        import huginn.hooks.science_hooks as mod

        hook = getattr(mod, hook_func)
        ctx = _make_ctx("some_other_tool", {"result": {"nan": True}})
        await hook(ctx)
        assert ctx.metadata.get("blocked_by_hook") is not True
        assert "warnings" not in ctx.metadata or len(ctx.metadata["warnings"]) == 0


# ── 6. 闭环验证: 主动学习迭代 ────────────────────────────────────


class TestClosedLoop:
    """验证主动学习闭环: gp → active_learning → gp."""

    def test_gp_suggests_active_learning(self):
        rules = [r for r in PIPELINE_RULES if r.tool_name == "gp_tool"]
        hints = set()
        for r in rules:
            hints.update(r.next_tool_hints)
        assert "active_learning_tool" in hints

    def test_active_learning_suggests_gp(self):
        rules = [r for r in PIPELINE_RULES if r.tool_name == "active_learning_tool"]
        hints = set()
        for r in rules:
            hints.update(r.next_tool_hints)
        assert "gp_tool" in hints
