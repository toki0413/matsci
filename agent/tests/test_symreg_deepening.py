"""Tests for SymbolicRegressionTool new actions: constraint_check + sobol_indices."""

from __future__ import annotations

import math

import numpy as np
import pytest

from huginn.tools.symbolic_regression_tool import (
    SymbolicRegressionInput,
    SymbolicRegressionTool,
)
from huginn.types import ToolContext


@pytest.fixture
def tool() -> SymbolicRegressionTool:
    return SymbolicRegressionTool()


@pytest.fixture
def ctx() -> ToolContext:
    return ToolContext(session_id="test", workspace=".")


class TestConstraintCheck:
    """Physical constraint priors on candidate expressions."""

    @pytest.mark.asyncio
    async def test_positivity_pass(self, tool, ctx):
        # y = x^2 >= 0 总是成立
        args = SymbolicRegressionInput(
            action="constraint_check",
            probe_expression="x**2",
            constraints={
                "positivity": True,
                "bounds": {"x": [-2.0, 2.0]},
            },
        )
        result = await tool.call(args, ctx)
        assert result.success, result.error
        pos_check = [c for c in result.data["checks"] if c["name"] == "positivity"][0]
        assert pos_check["passed"] is True
        assert result.data["all_passed"] is True

    @pytest.mark.asyncio
    async def test_positivity_fail(self, tool, ctx):
        # y = -x^2 在 x≠0 时 < 0
        args = SymbolicRegressionInput(
            action="constraint_check",
            probe_expression="-x**2",
            constraints={
                "positivity": True,
                "bounds": {"x": [-2.0, 2.0]},
            },
        )
        result = await tool.call(args, ctx)
        assert result.success
        pos_check = [c for c in result.data["checks"] if c["name"] == "positivity"][0]
        assert pos_check["passed"] is False
        assert pos_check["violations"] > 0
        assert result.data["all_passed"] is False

    @pytest.mark.asyncio
    async def test_monotonic_increasing(self, tool, ctx):
        # y = exp(x) 在 x 上单调递增
        args = SymbolicRegressionInput(
            action="constraint_check",
            probe_expression="exp(x)",
            constraints={
                "monotonic_in": ["x"],
                "bounds": {"x": [-1.0, 1.0]},
            },
        )
        result = await tool.call(args, ctx)
        assert result.success, result.error
        mono = [c for c in result.data["checks"] if "monotonic_in" in c["name"]][0]
        assert mono["passed"] is True

    @pytest.mark.asyncio
    async def test_monotonic_decreasing(self, tool, ctx):
        # y = 1/x 在 x > 0 时单调递减
        args = SymbolicRegressionInput(
            action="constraint_check",
            probe_expression="1/x",
            constraints={
                "monotonic_decreasing_in": ["x"],
                "bounds": {"x": [0.1, 5.0]},
            },
        )
        result = await tool.call(args, ctx)
        assert result.success
        mono = [c for c in result.data["checks"] if "monotonic_decreasing_in" in c["name"]][0]
        assert mono["passed"] is True

    @pytest.mark.asyncio
    async def test_finiteness_violation(self, tool, ctx):
        # y = 1/x 在 x=0 附近发散 → finiteness 失败
        args = SymbolicRegressionInput(
            action="constraint_check",
            probe_expression="1/x",
            constraints={
                "bounds": {"x": [-1.0, 1.0]},
            },
        )
        result = await tool.call(args, ctx)
        assert result.success
        finite = [c for c in result.data["checks"] if c["name"] == "finiteness"][0]
        assert finite["passed"] is False
        assert finite["n_inf"] > 0 or finite["n_nan"] > 0

    @pytest.mark.asyncio
    async def test_missing_expression(self, tool, ctx):
        args = SymbolicRegressionInput(
            action="constraint_check",
            constraints={"positivity": True, "bounds": {"x": [0, 1]}},
        )
        result = await tool.call(args, ctx)
        assert not result.success
        assert "probe_expression" in result.error


class TestSobolIndices:
    """Variance-based Sobol sensitivity analysis."""

    @pytest.mark.asyncio
    async def test_additive_model(self, tool, ctx):
        # y = x1 + x2; 两者方差相同 → S1 ≈ S2 ≈ 0.5, ST ≈ 0.5
        args = SymbolicRegressionInput(
            action="sobol_indices",
            sobol_model="x1 + x2",
            n_sobol_samples=2048,
            constraints={"bounds": {"x1": [0.0, 1.0], "x2": [0.0, 1.0]}},
        )
        result = await tool.call(args, ctx)
        assert result.success, result.error
        S = result.data["first_order"]
        # 容忍 Monte Carlo 误差 ±0.05
        assert abs(S["x1"] - 0.5) < 0.05, f"S1={S['x1']}"
        assert abs(S["x2"] - 0.5) < 0.05, f"S2={S['x2']}"
        # 加性模型无交互 → ST ≈ S
        ST = result.data["total"]
        assert abs(ST["x1"] - 0.5) < 0.06
        assert abs(ST["x2"] - 0.5) < 0.06

    @pytest.mark.asyncio
    async def test_dominant_feature(self, tool, ctx):
        # y = 10*x1 + 0.1*x2;  x1 占主导
        args = SymbolicRegressionInput(
            action="sobol_indices",
            sobol_model="10*x1 + 0.1*x2",
            n_sobol_samples=2048,
            constraints={"bounds": {"x1": [0.0, 1.0], "x2": [0.0, 1.0]}},
        )
        result = await tool.call(args, ctx)
        assert result.success
        ST = result.data["total"]
        assert ST["x1"] > ST["x2"]
        assert ST["x1"] > 0.9  # x1 主导
        ranking = result.data["ranking"]
        assert ranking[0]["feature"] == "x1"

    @pytest.mark.asyncio
    async def test_interaction(self, tool, ctx):
        # y = x1 * x2;  纯交互 → S_i 应小, ST_i 应大
        args = SymbolicRegressionInput(
            action="sobol_indices",
            sobol_model="x1 * x2",
            n_sobol_samples=4096,
            constraints={"bounds": {"x1": [0.0, 1.0], "x2": [0.0, 1.0]}},
        )
        result = await tool.call(args, ctx)
        assert result.success, result.error
        S = result.data["first_order"]
        ST = result.data["total"]
        # 纯乘积模型 S_i ≈ 0 (无主效应), ST_i ≈ 0.75 (含交互)
        # 但实际 Saltelli 一阶估计 S_i 也有非零值 (取决于具体估计器)
        # 这里只断言 ST > S (交互存在的标志)
        for f in ("x1", "x2"):
            assert ST[f] > S[f], f"ST({f})={ST[f]} should exceed S({f})={S[f]}"

    @pytest.mark.asyncio
    async def test_missing_bounds(self, tool, ctx):
        args = SymbolicRegressionInput(
            action="sobol_indices",
            sobol_model="x1 + x2",
        )
        result = await tool.call(args, ctx)
        assert not result.success
        assert "bounds" in result.error

    @pytest.mark.asyncio
    async def test_constant_model_zero_indices(self, tool, ctx):
        args = SymbolicRegressionInput(
            action="sobol_indices",
            sobol_model="5",
            n_sobol_samples=256,
            constraints={"bounds": {"x1": [0.0, 1.0]}},
        )
        result = await tool.call(args, ctx)
        assert result.success
        assert result.data["variance"] < 1e-12
        assert all(v == 0.0 for v in result.data["first_order"].values())
