"""Tests for skills modules."""

import asyncio

import pytest

from matsci_agent.skills.base import SkillDefinition, SkillParameter, SkillStep, DeclarativeSkillExecutor
from matsci_agent.skills.registry import SkillRegistry
from matsci_agent.skills.presets import STANDARD_DFT, SYMBOLIC_REGRESSION, SYMBOLIC_VERIFY


class TestSkillRegistry:
    def test_register_and_get(self):
        skill = SkillDefinition(
            name="test_skill",
            description="A test skill",
            category="test",
            parameters=[SkillParameter("x", "int", "input")],
        )
        SkillRegistry.register(skill)
        assert SkillRegistry.get("test_skill") == skill

    def test_search(self):
        results = SkillRegistry.search("dft")
        assert len(results) > 0
        assert any("dft" in s.name.lower() for s in results)

    def test_list_by_category(self):
        comp = SkillRegistry.get_by_category("computation")
        assert len(comp) > 0
        assert all(s.category == "computation" for s in comp)


class TestSkillDefinitions:
    def test_standard_dft_has_steps(self):
        assert len(STANDARD_DFT.steps) > 0
        assert "structure_file" in [p.name for p in STANDARD_DFT.parameters]

    def test_symbolic_regression_has_sr_tool(self):
        assert any(s.tool == "symbolic_regression_tool" for s in SYMBOLIC_REGRESSION.steps)

    def test_to_prompt(self):
        prompt = STANDARD_DFT.to_prompt()
        assert "Skill: standard_dft" in prompt
        assert "Parameters:" in prompt


class TestSymbolicVerifySkill:
    def test_skill_registered(self):
        from matsci_agent.skills.registry import SkillRegistry
        skill = SkillRegistry.get("symbolic_verify")
        assert skill is not None
        assert skill.category == "verification"

    def test_has_symbolic_and_lean_steps(self):
        assert len(SYMBOLIC_VERIFY.steps) == 2
        assert SYMBOLIC_VERIFY.steps[0].tool == "symbolic_math_tool"
        assert SYMBOLIC_VERIFY.steps[1].tool == "lean_tool"
        assert SYMBOLIC_VERIFY.steps[1].on_failure == "abort"

    def test_input_mapping_passes_symbolic_result(self):
        lean_step = SYMBOLIC_VERIFY.steps[1]
        assert lean_step.input_mapping["symbolic_result"] == "$symbolic_result"
        assert lean_step.input_mapping["action"] == "'auto_verify'"

    def test_required_tools(self):
        assert "symbolic_math_tool" in SYMBOLIC_VERIFY.required_tools
        assert "lean_tool" in SYMBOLIC_VERIFY.required_tools


class TestDeclarativeSkillExecutor:
    def test_missing_tool_returns_error(self):
        from matsci_agent.tools.registry import ToolRegistry
        executor = DeclarativeSkillExecutor(ToolRegistry)

        skill = SkillDefinition(
            name="bad_skill",
            description="uses missing tool",
            category="test",
            steps=[
                SkillStep(name="step1", tool="nonexistent_tool", input_mapping={}, output_key="out"),
            ],
        )
        import asyncio
        result = asyncio.run(executor.execute(skill, {}, {}))
        assert result["success"] is False
        assert result["steps"][0]["error"] == "Tool 'nonexistent_tool' not found"


class TestUQGPSkills:
    @staticmethod
    def _ensure_tools():
        from matsci_agent.tools.registry import ToolRegistry
        from matsci_agent.tools.uq_tool import UQTool
        from matsci_agent.tools.gp_tool import GPTool
        if "uq_tool" not in ToolRegistry.list_tools():
            ToolRegistry.register(UQTool())
        if "gp_tool" not in ToolRegistry.list_tools():
            ToolRegistry.register(GPTool())

    def test_uncertainty_propagation_skill_registered(self):
        from matsci_agent.skills.registry import SkillRegistry
        skill = SkillRegistry.get("uncertainty_propagation")
        assert skill is not None
        assert skill.category == "analysis"
        assert "uq_tool" in skill.required_tools

    def test_gp_prediction_skill_registered(self):
        from matsci_agent.skills.registry import SkillRegistry
        skill = SkillRegistry.get("gp_prediction")
        assert skill is not None
        assert "gp_tool" in skill.required_tools

    def test_bayesian_calibration_skill_registered(self):
        from matsci_agent.skills.registry import SkillRegistry
        skill = SkillRegistry.get("bayesian_calibration")
        assert skill is not None
        assert "gp_tool" in skill.required_tools

    def test_uncertainty_propagation_execution(self):
        self._ensure_tools()
        from matsci_agent.tools.registry import ToolRegistry
        from matsci_agent.skills.registry import SkillRegistry

        skill = SkillRegistry.get("uncertainty_propagation")
        executor = DeclarativeSkillExecutor(ToolRegistry)
        result = asyncio.run(executor.execute(
            skill,
            {
                "expression": "E * epsilon",
                "variables": [
                    {"name": "E", "distribution": "uniform", "low": 200e9, "high": 220e9},
                    {"name": "epsilon", "distribution": "normal", "mean": 0.001, "std": 0.0001},
                ],
                "n_samples": 500,
                "seed": 42,
            },
            {},
        ))
        assert result["success"] is True
        assert result["steps"][0]["output"]["mean"] > 0

    def test_gp_prediction_execution(self):
        self._ensure_tools()
        from matsci_agent.tools.registry import ToolRegistry
        from matsci_agent.skills.registry import SkillRegistry

        skill = SkillRegistry.get("gp_prediction")
        executor = DeclarativeSkillExecutor(ToolRegistry)
        result = asyncio.run(executor.execute(
            skill,
            {
                "X": [[0.0], [1.0], [2.0]],
                "y": [0.0, 1.0, 1.5],
                "X_new": [[0.5], [1.5]],
            },
            {},
        ))
        assert result["success"] is True
        assert "mean" in result["steps"][0]["output"]

    def test_bayesian_calibration_execution(self):
        self._ensure_tools()
        from matsci_agent.tools.registry import ToolRegistry
        from matsci_agent.skills.registry import SkillRegistry

        skill = SkillRegistry.get("bayesian_calibration")
        executor = DeclarativeSkillExecutor(ToolRegistry)
        result = asyncio.run(executor.execute(
            skill,
            {
                "objective_expression": "-(x - 2.5)**2 + 5",
                "calibration_variables": [{"name": "x", "low": 0.0, "high": 5.0}],
                "n_initial": 3,
                "n_iterations": 3,
                "maximize": True,
                "seed": 0,
            },
            {},
        ))
        assert result["success"] is True
        assert "best_X" in result["steps"][0]["output"]
