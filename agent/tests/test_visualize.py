"""Tests for the unified framework visualization layer."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from huginn.tools.symbolic_math_tool import SymbolicMathTool, SymbolicMathInput
from huginn.types import ToolContext
from huginn.unified.models import heat_equation_fem
from huginn.unified.visualize import plot_solution, solve_and_plot


CTX = ToolContext(session_id="test", workspace=".")


def test_plot_solution_creates_file() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "sol.png"
        result = plot_solution(
            mesh=[0.0, 0.5, 1.0],
            solution=[0.0, 0.25, 1.0],
            output_path=path,
            title="Test",
        )
        assert result == path
        assert path.exists()


def test_solve_and_plot() -> None:
    problem = heat_equation_fem(k=1.0, f=1.0)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "heat.png"
        result = solve_and_plot(problem, method="fem", n=8, output_path=path)
        assert "plot_path" in result
        assert Path(result["plot_path"]).exists()
        assert result["residual"] < 1e-10


class TestUnifiedSymbolicVisualize:
    @pytest.fixture
    def tool(self):
        return SymbolicMathTool()

    def test_unified_solve_and_plot(self, tool):
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_path = str(Path(tmpdir) / "unified.png")
            result = asyncio.run(tool.call(
                SymbolicMathInput(
                    action="unified",
                    target="solve_and_plot",
                    expression="heat_equation_fem",
                    variable="fem",
                    order=6,
                ),
                CTX,
            ))
            # Default output path is used when target is not supplied
            assert result.success
            assert "plot_path" in result.data
            assert Path(result.data["plot_path"]).exists()

    def test_unified_solve_and_plot_custom_path(self, tool):
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_path = str(Path(tmpdir) / "custom.png")
            result = asyncio.run(tool.call(
                SymbolicMathInput(
                    action="unified",
                    target="solve_and_plot",
                    expression="heat_equation_fem",
                    variable="fem",
                    order=4,
                    output_path=plot_path,
                ),
                CTX,
            ))
            assert result.success
            assert result.data["plot_path"] == plot_path
            assert Path(plot_path).exists()
