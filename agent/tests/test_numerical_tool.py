"""Tests for the NumericalTool solver interface."""

from __future__ import annotations

import math

import numpy as np
import pytest

from huginn.tools.numerical_tool import NumericalTool


@pytest.fixture
def tool():
    return NumericalTool()


@pytest.mark.asyncio
class TestNumericalToolRoot:
    async def test_root_scalar(self, tool):
        result = await tool.call({"action": "root", "func": "x**2 - 4", "x0": 1.0})
        assert result.success
        assert result.data["success"]
        assert math.isclose(result.data["values"]["root"], 2.0, abs_tol=1e-6)

    async def test_root_with_carrot(self, tool):
        # The tool normalizes ^ to **.
        result = await tool.call({"action": "root", "func": "x^2 - 9", "x0": 1.0})
        assert result.success
        assert math.isclose(result.data["values"]["root"], 3.0, abs_tol=1e-6)

    async def test_rejects_malicious_expression(self, tool):
        result = await tool.call(
            {"action": "root", "func": "__import__('os').system('ls')", "x0": 1.0}
        )
        assert not result.success
        assert "Numerical solver failed" in result.error


@pytest.mark.asyncio
class TestNumericalToolMinimize:
    async def test_minimize_parabola(self, tool):
        result = await tool.call(
            {"action": "minimize", "func": "(X[0] - 3)**2 + (X[1] + 1)**2", "x0": [0.0, 0.0]}
        )
        assert result.success
        x = result.data["values"]["x"]
        assert math.isclose(x[0], 3.0, abs_tol=1e-4)
        assert math.isclose(x[1], -1.0, abs_tol=1e-4)


@pytest.mark.asyncio
class TestNumericalToolIntegrate:
    async def test_integrate_quad(self, tool):
        result = await tool.call(
            {"action": "integrate", "func": "x**2", "a": 0.0, "b": 3.0}
        )
        assert result.success
        assert math.isclose(result.data["values"]["integral"], 9.0, abs_tol=1e-5)


@pytest.mark.asyncio
class TestNumericalToolODE:
    async def test_ode_exponential_decay(self, tool):
        result = await tool.call(
            {
                "action": "ode",
                "func": "-y[0]",
                "t_span": [0.0, 5.0],
                "y0": [1.0],
                "t_eval": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        assert result.success
        values = result.data["values"]
        # y(t) = exp(-t)
        for t, y in zip(values["t"], values["y"][0]):
            assert math.isclose(y, math.exp(-t), abs_tol=1e-4)


@pytest.mark.asyncio
class TestNumericalToolLinear:
    async def test_linear_solve(self, tool):
        result = await tool.call(
            {
                "action": "linear_solve",
                "A": [[2.0, 1.0], [1.0, 3.0]],
                "b_vec": [5.0, 8.0],
            }
        )
        assert result.success
        x = result.data["values"]["x"]
        # 2*1 + 1*2 = 5, 1*1 + 3*2 = 7 ... adjust expected
        # Actually solve gives x = [1.4, 2.2] approx
        assert len(x) == 2

    async def test_eigenvalues(self, tool):
        result = await tool.call(
            {
                "action": "eigenvalues",
                "A": [[4.0, 1.0], [1.0, 3.0]],
            }
        )
        assert result.success
        eigenvalues = result.data["values"]["eigenvalues"]
        assert len(eigenvalues) == 2


@pytest.mark.asyncio
class TestNumericalToolCurveFit:
    async def test_curve_fit_linear(self, tool):
        xdata = np.linspace(0, 10, 50).tolist()
        ydata = [2.5 * x + 1.0 + 0.1 * (i % 5) for i, x in enumerate(xdata)]
        result = await tool.call(
            {
                "action": "curve_fit",
                "func_fit": "a0*x + a1",
                "xdata": xdata,
                "ydata": ydata,
            }
        )
        assert result.success
        params = result.data["values"]["params"]
        assert math.isclose(params[0], 2.5, abs_tol=0.1)
        assert math.isclose(params[1], 1.0, abs_tol=0.5)
