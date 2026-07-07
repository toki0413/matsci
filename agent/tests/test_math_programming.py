"""Tests for LP / MILP / convex optimization in numerical_tool."""
from __future__ import annotations

import numpy as np
import pytest

from huginn.tools.numerical_tool import NumericalTool, NumericalToolInput
from huginn.types import ToolResult


def _run(tool: NumericalTool, args: dict) -> ToolResult:
    """Run the tool synchronously (call is async)."""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(tool.call(args))


@pytest.fixture
def tool():
    return NumericalTool()


# ── LP ──


class TestLP:
    def test_simple_lp(self, tool):
        # max x1 + x2  →  min -(x1 + x2)
        # s.t. x1 + x2 <= 4, x1 <= 3, x2 <= 3, x >= 0
        res = _run(tool, {
            "action": "lp",
            "c_vec": [-1, -1],
            "A_ub": [[1, 1], [1, 0], [0, 1]],
            "b_ub": [4, 3, 3],
        })
        assert res.success, res.error
        assert res.data["fun"] == pytest.approx(-4.0, abs=0.01)
        # x1 + x2 = 4
        assert sum(res.data["x"]) == pytest.approx(4.0, abs=0.01)

    def test_lp_with_equality(self, tool):
        # min x1 + 2*x2
        # s.t. x1 + x2 = 1, x >= 0
        res = _run(tool, {
            "action": "lp",
            "c_vec": [1, 2],
            "A_eq": [[1, 1]],
            "b_eq": [1],
        })
        assert res.success, res.error
        # Optimal: x1=1, x2=0, fun=1
        assert res.data["fun"] == pytest.approx(1.0, abs=0.01)
        assert res.data["x"][0] == pytest.approx(1.0, abs=0.01)

    def test_lp_bounds(self, tool):
        # min x1  s.t. x1 >= 2 (via bounds)
        res = _run(tool, {
            "action": "lp",
            "c_vec": [1],
            "var_bounds": [[2, 10]],
        })
        assert res.success, res.error
        assert res.data["x"][0] == pytest.approx(2.0, abs=0.01)

    def test_lp_missing_c(self, tool):
        res = _run(tool, {"action": "lp"})
        assert not res.success

    def test_lp_alloy_optimization(self, tool):
        """Classic alloy composition: max strength = 50x1 + 30x2 + 80x3
        s.t. x1+x2+x3=1, cost=10x1+20x2+5x3<=15, 0<=xi<=0.6
        """
        res = _run(tool, {
            "action": "lp",
            "c_vec": [-50, -30, -80],
            "A_eq": [[1, 1, 1]],
            "b_eq": [1],
            "A_ub": [[10, 20, 5]],
            "b_ub": [15],
            "var_bounds": [[0, 0.6], [0, 0.6], [0, 0.6]],
        })
        assert res.success, res.error
        x = res.data["x"]
        assert sum(x) == pytest.approx(1.0, abs=0.01)
        assert 10 * x[0] + 20 * x[1] + 5 * x[2] <= 15 + 0.01
        # All within bounds
        for xi in x:
            assert 0 <= xi <= 0.6 + 0.01


# ── MILP ──


class TestMILP:
    def test_simple_milp(self, tool):
        # min x1 + x2
        # s.t. x1 + x2 >= 3, x1, x2 non-negative integers
        # → x1=1, x2=2 or x1=2, x2=1 or x1=3,x2=0 etc, fun=3
        res = _run(tool, {
            "action": "milp",
            "c_vec": [1, 1],
            "A_ub": [[-1, -1]],  # -x1 - x2 <= -3
            "b_ub": [-3],
            "integrality": [1, 1],
            "var_bounds": [[0, 10], [0, 10]],
        })
        assert res.success, res.error
        assert sum(res.data["x"]) == pytest.approx(3.0, abs=0.5)
        # Should be integers
        for xi in res.data["x"]:
            assert xi == pytest.approx(int(xi), abs=0.01)

    def test_milp_experiment_scheduling(self, tool):
        """Select which experiments to run: min cost, max info gain.
        min 10x1 + 20x2 + 5x3
        s.t. 5x1 + 8x2 + 3x3 >= 8 (info gain threshold)
             x1 + x2 + x3 <= 2 (budget: max 2 experiments)
             xi in {0, 1}
        """
        res = _run(tool, {
            "action": "milp",
            "c_vec": [10, 20, 5],
            "A_ub": [[-5, -8, -3], [1, 1, 1]],
            "b_ub": [-8, 2],
            "integrality": [1, 1, 1],
            "var_bounds": [[0, 1], [0, 1], [0, 1]],
        })
        assert res.success, res.error
        x = res.data["x"]
        # All binary
        for xi in x:
            assert xi in (0, 1) or xi == pytest.approx(0, abs=0.01) or xi == pytest.approx(1, abs=0.01)
        # Info gain constraint satisfied
        assert 5 * x[0] + 8 * x[1] + 3 * x[2] >= 8 - 0.01
        # Budget constraint
        assert sum(x) <= 2 + 0.01

    def test_milp_missing_c(self, tool):
        res = _run(tool, {"action": "milp"})
        assert not res.success


# ── Convex ──


class TestConvex:
    @pytest.fixture(autouse=True)
    def _skip_if_no_cvxpy(self):
        pytest.importorskip("cvxpy")

    def test_simple_least_squares(self, tool):
        """minimize sum_squares(A@x - b) subject to x >= 0."""
        # Over-determined system: x = [1, 2]
        A = [[1, 0], [0, 1], [1, 1]]
        b = [1, 2, 3]
        res = _run(tool, {
            "action": "convex",
            "convex_problem": "minimize sum_squares(A @ x - b)\nsubject to\nx >= 0",
            "A_ub": A,
            "b_ub": b,
        })
        assert res.success, res.error
        assert res.data["x"][0] == pytest.approx(1.0, abs=0.1)
        assert res.data["x"][1] == pytest.approx(2.0, abs=0.1)

    def test_convex_missing_problem(self, tool):
        res = _run(tool, {"action": "convex"})
        assert not res.success

    def test_cvxpy_not_installed(self, tool, monkeypatch):
        """When cvxpy is not installed, return a helpful error."""
        pytest.importorskip("cvxpy")  # skip if cvxpy genuinely unavailable
        import builtins
        orig_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "cvxpy":
                raise ImportError("No module named 'cvxpy'")
            return orig_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        res = _run(tool, {
            "action": "convex",
            "convex_problem": "minimize sum_squares(x)",
        })
        assert not res.success
        assert "cvxpy" in res.error.lower()
