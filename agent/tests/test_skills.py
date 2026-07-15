"""Tests for skills modules."""

import asyncio

from huginn.skills.base import (
    DeclarativeSkillExecutor,
    SkillDefinition,
    SkillParameter,
    SkillStep,
)
from huginn.skills.presets import STANDARD_DFT, SYMBOLIC_REGRESSION, SYMBOLIC_VERIFY
from huginn.skills.registry import SkillRegistry


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
        assert any(
            s.tool == "symbolic_regression_tool" for s in SYMBOLIC_REGRESSION.steps
        )

    def test_to_prompt(self):
        prompt = STANDARD_DFT.to_prompt()
        assert "Skill: standard_dft" in prompt
        assert "Parameters:" in prompt


class TestSymbolicVerifySkill:
    def test_skill_registered(self):
        from huginn.skills.registry import SkillRegistry

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
        from huginn.tools.registry import ToolRegistry

        executor = DeclarativeSkillExecutor(ToolRegistry)

        skill = SkillDefinition(
            name="bad_skill",
            description="uses missing tool",
            category="test",
            steps=[
                SkillStep(
                    name="step1",
                    tool="nonexistent_tool",
                    input_mapping={},
                    output_key="out",
                ),
            ],
        )
        import asyncio

        result = asyncio.run(executor.execute(skill, {}, {}))
        assert result["success"] is False
        assert result["steps"][0]["error"] == "Tool 'nonexistent_tool' not found"

    def test_parallel_group_runs_concurrently(self):
        """同 parallel_group 的 step 应并行执行 (耗时 ≈ max 而非 sum)."""
        import asyncio
        import time

        from huginn.tools.base import HuginnTool, ToolResult
        from huginn.tools.registry import ToolRegistry
        from pydantic import BaseModel

        # 清理残留注册, 避免上一次测试污染
        ToolRegistry.clear()

        class SlowInput(BaseModel):
            label: str = ""
            delay: float = 0.2

        # 记录每个 step 的启动时间, 验证并发
        start_times: dict[str, float] = {}

        class SlowTool(HuginnTool):
            input_schema = SlowInput
            read_only = True

            async def call(self, raw, ctx):
                data = raw if isinstance(raw, dict) else raw.model_dump()
                start_times[data["label"]] = time.monotonic()
                await asyncio.sleep(data["delay"])
                return ToolResult(success=True, data={"label": data["label"]})

        class SlowToolA(SlowTool):
            name = "slow_a"
            description = "slow tool a"

        class SlowToolB(SlowTool):
            name = "slow_b"
            description = "slow tool b"

        ToolRegistry.register(SlowToolA())
        ToolRegistry.register(SlowToolB())

        executor = DeclarativeSkillExecutor(ToolRegistry)
        skill = SkillDefinition(
            name="parallel_test",
            description="two parallel slow steps",
            category="test",
            steps=[
                SkillStep(
                    name="step_a",
                    tool="slow_a",
                    input_mapping={"label": "'a'", "delay": "0.2"},
                    output_key="out_a",
                    parallel_group="g1",
                ),
                SkillStep(
                    name="step_b",
                    tool="slow_b",
                    input_mapping={"label": "'b'", "delay": "0.2"},
                    output_key="out_b",
                    parallel_group="g1",
                ),
            ],
        )

        t0 = time.monotonic()
        result = asyncio.run(executor.execute(skill, {}, {}))
        elapsed = time.monotonic() - t0

        assert result["success"] is True
        # 串行 ≈ 0.4s, 并行 ≈ 0.2s. 用 0.35s 阈值区分
        assert elapsed < 0.35, f"expected parallel (<0.35s), got {elapsed:.3f}s"
        # 两个 step 启动时间差应 < delay (基本同时启动)
        if "a" in start_times and "b" in start_times:
            assert abs(start_times["a"] - start_times["b"]) < 0.1

        ToolRegistry.clear()


class TestUQGPSkills:
    @staticmethod
    def _ensure_tools():
        from huginn.tools.gp_tool import GPTool
        from huginn.tools.registry import ToolRegistry
        from huginn.tools.uq_tool import UQTool

        if "uq_tool" not in ToolRegistry.list_tools():
            ToolRegistry.register(UQTool())
        if "gp_tool" not in ToolRegistry.list_tools():
            ToolRegistry.register(GPTool())

    def test_uncertainty_propagation_skill_registered(self):
        from huginn.skills.registry import SkillRegistry

        skill = SkillRegistry.get("uncertainty_propagation")
        assert skill is not None
        assert skill.category == "analysis"
        assert "uq_tool" in skill.required_tools

    def test_gp_prediction_skill_registered(self):
        from huginn.skills.registry import SkillRegistry

        skill = SkillRegistry.get("gp_prediction")
        assert skill is not None
        assert "gp_tool" in skill.required_tools

    def test_bayesian_calibration_skill_registered(self):
        from huginn.skills.registry import SkillRegistry

        skill = SkillRegistry.get("bayesian_calibration")
        assert skill is not None
        assert "gp_tool" in skill.required_tools

    def test_uncertainty_propagation_execution(self):
        self._ensure_tools()
        from huginn.skills.registry import SkillRegistry
        from huginn.tools.registry import ToolRegistry

        skill = SkillRegistry.get("uncertainty_propagation")
        executor = DeclarativeSkillExecutor(ToolRegistry)
        result = asyncio.run(
            executor.execute(
                skill,
                {
                    "expression": "E * epsilon",
                    "variables": [
                        {
                            "name": "E",
                            "distribution": "uniform",
                            "low": 200e9,
                            "high": 220e9,
                        },
                        {
                            "name": "epsilon",
                            "distribution": "normal",
                            "mean": 0.001,
                            "std": 0.0001,
                        },
                    ],
                    "n_samples": 500,
                    "seed": 42,
                },
                {},
            )
        )
        assert result["success"] is True
        assert result["steps"][0]["output"]["mean"] > 0

    def test_gp_prediction_execution(self):
        self._ensure_tools()
        from huginn.skills.registry import SkillRegistry
        from huginn.tools.registry import ToolRegistry

        skill = SkillRegistry.get("gp_prediction")
        executor = DeclarativeSkillExecutor(ToolRegistry)
        result = asyncio.run(
            executor.execute(
                skill,
                {
                    "X": [[0.0], [1.0], [2.0]],
                    "y": [0.0, 1.0, 1.5],
                    "X_new": [[0.5], [1.5]],
                },
                {},
            )
        )
        assert result["success"] is True
        assert "mean" in result["steps"][0]["output"]

    def test_bayesian_calibration_execution(self):
        self._ensure_tools()
        from huginn.skills.registry import SkillRegistry
        from huginn.tools.registry import ToolRegistry

        skill = SkillRegistry.get("bayesian_calibration")
        executor = DeclarativeSkillExecutor(ToolRegistry)
        result = asyncio.run(
            executor.execute(
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
            )
        )
        assert result["success"] is True
        assert "best_X" in result["steps"][0]["output"]


class TestSkillInputResolution:
    def test_resolve_context_reference(self):
        from huginn.skills.base import DeclarativeSkillExecutor

        assert DeclarativeSkillExecutor._resolve_value("$x", {"x": 42}) == 42

    def test_resolve_dotted_path(self):
        from huginn.skills.base import DeclarativeSkillExecutor

        context = {"result": {"data": {"value": 7}}}
        assert (
            DeclarativeSkillExecutor._resolve_value("$result.data.value", context) == 7
        )

    def test_resolve_literal_string(self):
        from huginn.skills.base import DeclarativeSkillExecutor

        assert DeclarativeSkillExecutor._resolve_value("'relax'", {}) == "relax"

    def test_resolve_literal_int(self):
        from huginn.skills.base import DeclarativeSkillExecutor

        assert DeclarativeSkillExecutor._resolve_value("3", {}) == 3

    def test_resolve_missing_path_returns_none(self):
        from huginn.skills.base import DeclarativeSkillExecutor

        assert DeclarativeSkillExecutor._resolve_value("$missing.key", {}) is None


class TestNewSkills:
    def test_topological_geometry_analysis_registered(self):
        from huginn.skills.registry import SkillRegistry

        skill = SkillRegistry.get("topological_geometry_analysis")
        assert skill is not None
        assert skill.category == "analysis"
        assert "descriptor_tool" in skill.required_tools
        assert "report_tool" in skill.required_tools

    def test_visualize_results_registered(self):
        from huginn.skills.registry import SkillRegistry

        skill = SkillRegistry.get("visualize_results")
        assert skill is not None
        assert skill.category == "reporting"
        assert "visualize_tool" in skill.required_tools

    def test_visualize_results_skill_runs(self, tmp_path):
        import json

        from huginn.config import HuginnConfig
        from huginn.skills.base import DeclarativeSkillExecutor
        from huginn.skills.registry import SkillRegistry
        from huginn.tools import register_all_tools
        from huginn.tools.registry import ToolRegistry

        ToolRegistry.clear()
        register_all_tools(config=HuginnConfig())

        report_path = tmp_path / "benchmark.json"
        report_path.write_text(
            json.dumps(
                {
                    "results": [
                        {
                            "task_id": "t1",
                            "category": "dft",
                            "passed": True,
                            "exec_time_seconds": 1.0,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        output_path = tmp_path / "figure.png"

        skill = SkillRegistry.get("visualize_results")
        executor = DeclarativeSkillExecutor(ToolRegistry)
        result = asyncio.run(
            executor.execute(
                skill,
                {
                    "report_path": str(report_path),
                    "action": "benchmark",
                    "output_path": str(output_path),
                    "plot_type": "pie",
                },
                {},
            )
        )

        assert result["success"] is True
        assert output_path.exists()
        ToolRegistry.clear()

    def test_topological_geometry_skill_runs(self, tmp_path):
        from huginn.config import HuginnConfig
        from huginn.skills.base import DeclarativeSkillExecutor
        from huginn.skills.registry import SkillRegistry
        from huginn.tools import register_all_tools
        from huginn.tools.registry import ToolRegistry

        ToolRegistry.clear()
        register_all_tools(config=HuginnConfig())

        output_path = tmp_path / "report.md"
        skill = SkillRegistry.get("topological_geometry_analysis")
        executor = DeclarativeSkillExecutor(ToolRegistry)
        result = asyncio.run(
            executor.execute(
                skill,
                {
                    "formula": "SiO2",
                    "output_path": str(output_path),
                },
                {},
            )
        )

        assert result["success"] is True
        assert output_path.exists()
        ToolRegistry.clear()

    def test_synthesis_planning_skill_runs(self, tmp_path):
        import csv

        import numpy as np

        from huginn.config import HuginnConfig
        from huginn.skills.base import DeclarativeSkillExecutor
        from huginn.skills.registry import SkillRegistry
        from huginn.tools import register_all_tools
        from huginn.tools.registry import ToolRegistry

        ToolRegistry.clear()
        register_all_tools(config=HuginnConfig())

        data_path = tmp_path / "synthesis.csv"
        rng = np.random.default_rng(42)
        rows = []
        for _ in range(10):
            temp = float(rng.uniform(300, 800))
            time = float(rng.uniform(1, 10))
            rows.append(
                {
                    "temperature": temp,
                    "time": time,
                    "yield": temp * 0.01 + time * 0.5 + rng.normal(0, 0.1),
                }
            )
        with data_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["temperature", "time", "yield"])
            writer.writeheader()
            writer.writerows(rows)

        output_path = tmp_path / "recommendations.md"
        skill = SkillRegistry.get("synthesis_planning")
        executor = DeclarativeSkillExecutor(ToolRegistry)
        result = asyncio.run(
            executor.execute(
                skill,
                {
                    "data_path": str(data_path),
                    "target_column": "yield",
                    "feature_columns": ["temperature", "time"],
                    "bounds": {"temperature": [300.0, 800.0], "time": [1.0, 10.0]},
                    "n_recommendations": 2,
                    "maximize": True,
                    "output_path": str(output_path),
                },
                {},
            )
        )

        assert result["success"] is True
        assert output_path.exists()
        ToolRegistry.clear()
