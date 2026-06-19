"""Unit tests for Huginn tools."""

import asyncio

from huginn.tools.adapter import ToolAdapter
from huginn.tools.diagnose_tool import DiagnoseTool
from huginn.tools.registry import ToolRegistry
from huginn.tools.structure_tool import StructureTool
from huginn.tools.validate_tool import ValidateTool
from huginn.types import ToolContext


class TestToolRegistry:
    def test_register_and_list(self):
        ToolRegistry.clear()
        ToolRegistry.register(StructureTool())
        assert "structure_tool" in ToolRegistry.list_tools()

    def test_get_schema(self):
        ToolRegistry.clear()
        ToolRegistry.register(StructureTool())
        schemas = ToolRegistry.get_all_schemas()
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "structure_tool"


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
        lc_tool = ToolAdapter.adapt(tool)
        assert lc_tool.name == "structure_tool"
        assert lc_tool.args_schema is not None
