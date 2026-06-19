"""Tests for the unified framework discretization layer."""

from __future__ import annotations

import asyncio

import pytest

from huginn.tools.symbolic_math_tool import SymbolicMathInput, SymbolicMathTool
from huginn.types import ToolContext
from huginn.unified import discretize
from huginn.unified.models import heat_equation_fem

CTX = ToolContext(session_id="test", workspace=".")


def test_fem_heat_discretization() -> None:
    problem = heat_equation_fem(k=2.0, f=1.0)
    result = discretize(problem, method="fem", n=4)
    assert result["method"] == "fem"
    assert result["n_dof"] == 5
    K = result["stiffness_matrix"]
    F = result["load_vector"]
    # 1D linear Laplacian stiffness is tridiagonal with 2/h and -/h
    assert K[0][0] > 0
    assert K[0][1] < 0
    assert sum(F) == pytest.approx(1.0, abs=1e-9)


def test_fd_heat_discretization() -> None:
    problem = heat_equation_fem(k=1.0, f=0.0)
    result = discretize(problem, method="fd", n=5)
    assert result["method"] == "fd"
    assert result["n_dof"] == 5
    A = result["stiffness_matrix"]
    # Boundary rows are identity
    assert A[0][0] == 1.0
    assert A[-1][-1] == 1.0
    # Interior row: [-c, 2c, -c]
    assert A[2][1] == A[2][3]
    assert A[2][1] < 0
    assert A[2][2] == -2.0 * A[2][1]


def test_discretize_unsupported_principle() -> None:
    from huginn.unified.models import harmonic_oscillator_md

    problem = harmonic_oscillator_md()
    with pytest.raises(ValueError, match="variational principles"):
        discretize(problem, method="fem", n=4)


class TestUnifiedSymbolicDiscretize:
    @pytest.fixture
    def tool(self):
        return SymbolicMathTool()

    def test_unified_discretize_fem(self, tool):
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(
                    action="unified",
                    target="discretize",
                    expression="linear_elasticity_fem",
                    variable="fem",
                    order=4,
                ),
                CTX,
            )
        )
        assert result.success
        assert result.data["method"] == "fem"
        assert result.data["n_dof"] == 5
        assert len(result.data["stiffness_matrix"]) == 5

    def test_unified_discretize_fd(self, tool):
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(
                    action="unified",
                    target="discretize",
                    expression="heat_equation_fem",
                    variable="fd",
                    order=5,
                ),
                CTX,
            )
        )
        assert result.success
        assert result.data["method"] == "fd"
        assert result.data["n_dof"] == 5
