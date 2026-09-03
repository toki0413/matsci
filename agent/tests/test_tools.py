"""Unit tests for Huginn tools."""

import asyncio

import pytest

from huginn.core_types import ToolContext
from huginn.tools.adapter import ToolAdapter
from huginn.tools.diagnose_tool import DiagnoseTool
from huginn.tools.registry import ToolRegistry
from huginn.tools.structure_tool import StructureTool
from huginn.tools.validate_tool import ValidateTool


@pytest.fixture(autouse=True)
def _isolate_registry():
    before = ToolRegistry.snapshot()
    yield
    ToolRegistry.restore(before)


class TestToolRegistry:
    def test_register_and_list(self):
        ToolRegistry.register(StructureTool())
        assert "structure_tool" in ToolRegistry.list_tools()

    def test_get_schema(self):
        # Registry is pre-populated by the session-level canonical baseline in
        # conftest, so assert on the newly registered tool's presence rather
        # than a total count of 1 (which only held when the registry was empty).
        ToolRegistry.register(StructureTool())
        schemas = ToolRegistry.get_all_schemas()
        by_name = {s["function"]["name"]: s for s in schemas}
        assert "structure_tool" in by_name
        assert by_name["structure_tool"]["function"]["name"] == "structure_tool"


class TestStructureTool:
    def test_nonexistent_file(self):
        tool = StructureTool()
        result = asyncio.run(
            tool.call(
                tool.input_schema(action="read", file_path="/nonexistent/POSCAR"),
                ToolContext(session_id="test", workspace="."),
            )
        )
        assert not result.success
        assert "not found" in result.error.lower()


class TestValidateTool:
    def test_dft_validation_pass(self):
        tool = ValidateTool()
        result = asyncio.run(
            tool.call(
                tool.input_schema(
                    result_type="dft",
                    result_data={"energy": -100.0, "max_force": 0.005, "band_gap": 1.5},
                ),
                ToolContext(session_id="test", workspace="."),
            )
        )
        assert result.success
        assert result.data["all_passed"]

    def test_dft_validation_fail(self):
        tool = ValidateTool()
        result = asyncio.run(
            tool.call(
                tool.input_schema(
                    result_type="dft",
                    result_data={"energy": 50.0, "max_force": 0.1, "band_gap": -0.5},
                ),
                ToolContext(session_id="test", workspace="."),
            )
        )
        assert result.success
        assert not result.data["all_passed"]

    def test_aggregate_physics_score_folds_world_reward(self):
        """阶段4: 世界预测命中奖励并进物理校验 r_phys; 缺省零回归."""
        tool = ValidateTool()
        checks = [
            {"name": "a", "passed": True, "severity": "error", "score": None},
            {"name": "b", "passed": False, "severity": "warn", "score": None},
        ]
        # 无 world_reward → 行为不变 (基分 = (1*3 + 0*2)/5 = 0.6)
        base = tool._aggregate_physics_score(checks)
        assert base == pytest.approx(0.6)
        # 有 world_reward (预测全中≈1) → 0.5*0.6 + 0.5*1 = 0.8
        folded = tool._aggregate_physics_score(checks, world_reward=1.0)
        assert folded == pytest.approx(0.8)


class TestDiagnoseTool:
    def test_vasp_eddav(self):
        tool = DiagnoseTool()
        result = asyncio.run(
            tool.call(
                tool.input_schema(
                    error_message="ERROR EDDDAV: Call to ZHEGV failed",
                    software="vasp",
                    calculation_type="DFT",
                ),
                ToolContext(session_id="test", workspace="."),
            )
        )
        assert result.success


class TestLangChainAdapter:
    def test_adapt_structure_tool(self):
        tool = StructureTool()
        lc_tool = ToolAdapter().adapt(tool)
        assert lc_tool.name == "structure_tool"
        assert lc_tool.args_schema is not None
