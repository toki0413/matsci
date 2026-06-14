"""Tests for the unified framework solver layer."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from huginn.tools.symbolic_math_tool import SymbolicMathTool, SymbolicMathInput
from huginn.types import ToolContext
from huginn.unified import solve
from huginn.unified.models import heat_equation_fem, linear_elasticity_fem


CTX = ToolContext(session_id="test", workspace=".")


def test_solve_heat_fem() -> None:
    problem = heat_equation_fem(k=1.0, f=1.0)
    result = solve(problem, method="fem", n=8)
    assert result["method"] == "fem"
    assert result["n_dof"] == 9
    assert len(result["solution"]) == 9
    assert result["residual"] < 1e-12
    # With unit source and zero boundary, interior solution is positive.
    assert all(v >= 0 for v in result["solution"])


def test_solve_elasticity_fd() -> None:
    problem = linear_elasticity_fem(E=2.0, f=0.0)
    result = solve(problem, method="fd", n=5)
    assert result["method"] == "fd"
    assert result["n_dof"] == 5
    assert len(result["solution"]) == 5
    assert result["residual"] < 1e-12
    # Zero source and zero Dirichlet -> zero solution.
    assert np.allclose(result["solution"], 0.0)


def test_solve_unknown_method() -> None:
    problem = heat_equation_fem()
    with pytest.raises(ValueError, match="Unknown discretization method"):
        solve(problem, method="spectral", n=4)


class TestUnifiedSymbolicSolve:
    @pytest.fixture
    def tool(self):
        return SymbolicMathTool()

    def test_unified_solve_fem(self, tool):
        result = asyncio.run(tool.call(
            SymbolicMathInput(
                action="unified",
                target="solve",
                expression="linear_elasticity_fem",
                variable="fem",
                order=6,
            ),
            CTX,
        ))
        assert result.success
        assert result.data["method"] == "fem"
        assert result.data["n_dof"] == 7
        assert "solution" in result.data
        assert result.data["residual"] < 1e-10

    def test_unified_solve_fd(self, tool):
        result = asyncio.run(tool.call(
            SymbolicMathInput(
                action="unified",
                target="solve",
                expression="heat_equation_fem",
                variable="fd",
                order=5,
            ),
            CTX,
        ))
        assert result.success
        assert result.data["method"] == "fd"
        assert result.data["n_dof"] == 5
        assert "solution" in result.data
