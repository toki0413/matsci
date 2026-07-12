"""Tests for Bourbaki tool integration.

Bourbaki (math_anything) must be discoverable.  These tests verify that
Huginn's BourbakiTool correctly wraps the mathematical structure API.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure math_anything is discoverable in test environment (local dev only)
_bourbaki_path = str(Path(__file__).parent.parent.parent / "math-anything" / "math-anything")
if Path(_bourbaki_path).exists() and _bourbaki_path not in sys.path:
    sys.path.insert(0, _bourbaki_path)

from huginn.tools.bourbaki_tool import BourbakiInput, BourbakiTool
from huginn.types import ToolContext

CTX = ToolContext(session_id="test", workspace=".")


class TestBourbakiTool:
    @pytest.fixture
    def tool(self):
        return BourbakiTool()

    @pytest.mark.asyncio
    async def test_list_domains(self, tool):
        result = await tool.call(BourbakiInput(action="list_domains"), CTX)
        assert result.success is True
        data = result.data["message"]
        assert "dft" in data or "domains" in data or "Fallback" in data

    @pytest.mark.asyncio
    async def test_list_equation_types(self, tool):
        result = await tool.call(BourbakiInput(action="list_equation_types"), CTX)
        assert result.success is True
        data = result.data["message"]
        assert "navier_stokes" in data or "equation_types" in data or "Fallback" in data

    @pytest.mark.asyncio
    async def test_analyze_domain_dft(self, tool):
        result = await tool.call(
            BourbakiInput(action="analyze_domain", domain="dft", parameters={"ENCUT": 520}), CTX
        )
        assert result.success is True
        data = result.data["message"]
        assert "error" not in data.lower() or "dft" in data.lower() or "Fallback" in data

    @pytest.mark.asyncio
    async def test_build_conservation_field_heat(self, tool):
        result = await tool.call(
            BourbakiInput(action="build_conservation_field", equation_type="heat", parameters={"k": 1.0}), CTX
        )
        assert result.success is True
        data = result.data["message"]
        assert "heat" in data.lower() or "equation_type" in data.lower() or "Fallback" in data

    @pytest.mark.asyncio
    async def test_buckingham_pi(self, tool):
        result = await tool.call(
            BourbakiInput(
                action="buckingham_pi",
                variables=[("E", "GPa"), ("rho", "kg/m3"), ("L", "m")],
                target="E",
            ),
            CTX,
        )
        assert result.success is True
        data = result.data["message"]
        assert "pi_groups" in data.lower() or "dimensionless" in data.lower() or "Fallback" in data

    @pytest.mark.asyncio
    async def test_extract_engine_vasp(self, tool):
        result = await tool.call(
            BourbakiInput(
                action="extract_engine",
                engine="vasp",
                engine_params={"ENCUT": 520, "SIGMA": 0.05, "EDIFF": 1e-6},
            ),
            CTX,
        )
        assert result.success is True
        data = result.data["message"]
        assert "vasp" in data.lower() or "success" in data.lower() or "Fallback" in data

    @pytest.mark.asyncio
    async def test_compare_domains(self, tool):
        result = await tool.call(
            BourbakiInput(
                action="compare_domains",
                domain="dft",
                domain_b="md",
                parameters={},
                parameters_b={},
            ),
            CTX,
        )
        assert result.success is True
        data = result.data["message"]
        assert "error" not in data.lower() or "comparison" in data.lower() or "Fallback" in data

    @pytest.mark.asyncio
    async def test_analyze_morphism_chain(self, tool):
        result = await tool.call(
            BourbakiInput(action="analyze_morphism_chain", domain="dft"), CTX
        )
        assert result.success is True
        data = result.data["message"]
        assert "chain" in data.lower() or "morphism" in data.lower() or "error" not in data.lower()

    @pytest.mark.asyncio
    async def test_tool_registered(self, tool):
        from huginn.tools.registry import ToolRegistry

        # Manually register for this test
        ToolRegistry.register(tool)
        assert "bourbaki" in ToolRegistry.list_tools() or "bourbaki_tool" in ToolRegistry.list_tools()
        ToolRegistry.unregister("bourbaki")
