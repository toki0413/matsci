"""Tests for SymbolicMathTool differential geometry actions + derive alias.

Covers: metric (Christoffel + Ricci), geodesic, curvature (Gaussian/mean),
lie_derivative (Lie bracket), connection (Levi-Civita), and derive → EL alias.
"""

from __future__ import annotations

import pytest
import sympy as sp

from huginn.tools.symbolic_math.tool import SymbolicMathInput, SymbolicMathTool
from huginn.types import ToolContext


@pytest.fixture
def tool() -> SymbolicMathTool:
    return SymbolicMathTool()


@pytest.fixture
def ctx() -> ToolContext:
    return ToolContext(session_id="test", workspace=".")


class TestDiffgeoMetric:
    """度规 → Christoffel 记号 + Ricci."""

    @pytest.mark.asyncio
    async def test_flat_2d_christoffel_zero(self, tool, ctx):
        # 平直欧氏度规 g = diag(1, 1), Christoffel 全零
        args = SymbolicMathInput(
            action="diffgeo_metric",
            target="christoffel",
            symbols=["x", "y"],
            matrix=[["1", "0"], ["0", "1"]],
        )
        result = await tool.call(args, ctx)
        assert result.success, result.error
        assert result.data["n_nonzero"] == 0

    @pytest.mark.asyncio
    async def test_polar_metric(self, tool, ctx):
        # 极坐标度规 ds^2 = dr^2 + r^2 dθ^2;  Γ^r_{θθ} = -r, Γ^θ_{rθ} = 1/r
        args = SymbolicMathInput(
            action="diffgeo_metric",
            target="christoffel",
            symbols=["r", "theta"],
            matrix=[["1", "0"], ["0", "r**2"]],
        )
        result = await tool.call(args, ctx)
        assert result.success
        nonzero = result.data["christoffel_nonzero"]
        # 至少 2 个非零 (Γ^r_{θθ} = -r, Γ^θ_{rθ} = 1/r)
        assert result.data["n_nonzero"] >= 2
        # 找到 Γ^r_{θθ}
        found = [n for n in nonzero if "r" in n["index"] and "theta" in n["index"]]
        assert len(found) >= 1

    @pytest.mark.asyncio
    async def test_flat_scalar_curvature_zero(self, tool, ctx):
        # 平直度规 Ricci 标量 R = 0
        args = SymbolicMathInput(
            action="diffgeo_metric",
            target="scalar",
            symbols=["x", "y"],
            matrix=[["1", "0"], ["0", "1"]],
        )
        result = await tool.call(args, ctx)
        assert result.success
        assert result.data["ricci_scalar"] == "0"


class TestDiffgeoGeodesic:
    """测地线方程."""

    @pytest.mark.asyncio
    async def test_flat_geodesic_straight(self, tool, ctx):
        # 平直 2D: 测地线方程是 d²x/ds² = 0, d²y/ds² = 0
        args = SymbolicMathInput(
            action="diffgeo_geodesic",
            symbols=["x", "y"],
            matrix=[["1", "0"], ["0", "1"]],
        )
        result = await tool.call(args, ctx)
        assert result.success, result.error
        eqs = result.data["geodesic_equations"]
        assert len(eqs) == 2
        # 平直情形下方程右端应为 0
        assert "0" in eqs[0]["equation"]

    @pytest.mark.asyncio
    async def test_sphere_geodesic_nontrivial(self, tool, ctx):
        # 单位球面 ds^2 = dθ^2 + sin²θ dφ^2;  测地线有非零 Γ 项
        args = SymbolicMathInput(
            action="diffgeo_geodesic",
            symbols=["theta", "phi"],
            matrix=[["1", "0"], ["0", "sin(theta)**2"]],
        )
        result = await tool.call(args, ctx)
        assert result.success
        eqs = result.data["geodesic_equations"]
        assert len(eqs) == 2
        # θ 方向方程应含 sin(θ)cos(θ) 项 (来自 Γ^θ_{φφ} = -sinθ cosθ)
        theta_eq = eqs[0]["equation"]
        assert "sin" in theta_eq


class TestDiffgeoCurvature:
    """曲面高斯/平均曲率."""

    @pytest.mark.asyncio
    async def test_plane_zero_curvature(self, tool, ctx):
        # 平面 r(u,v) = (u, v, 0);  K = 0, H = 0
        args = SymbolicMathInput(
            action="diffgeo_curvature",
            expression="u;v;0",
            symbols=["u", "v"],
        )
        result = await tool.call(args, ctx)
        assert result.success, result.error
        assert result.data["gaussian_curvature"] == "0"
        assert result.data["mean_curvature"] == "0"

    @pytest.mark.asyncio
    async def test_sphere_curvature_positive(self, tool, ctx):
        # 单位球面 r(u,v) = (sin(u)cos(v), sin(u)sin(v), cos(u));  K = 1, H = -1 (或 +1 取决于法向)
        args = SymbolicMathInput(
            action="diffgeo_curvature",
            expression="sin(u)*cos(v);sin(u)*sin(v);cos(u)",
            symbols=["u", "v"],
        )
        result = await tool.call(args, ctx)
        assert result.success
        K = sp.sympify(result.data["gaussian_curvature"])
        assert sp.simplify(K - 1) == 0


class TestDiffgeoLieDerivative:
    """李导数 = 李括号."""

    @pytest.mark.asyncio
    async def test_bracket_commuting_fields_zero(self, tool, ctx):
        # X = ∂_x, Y = ∂_y → [X, Y] = 0
        args = SymbolicMathInput(
            action="diffgeo_lie_derivative",
            symbols=["x", "y"],
            matrix=[["1"], ["0"]],
            expression="0;1",
        )
        result = await tool.call(args, ctx)
        assert result.success, result.error
        assert result.data["lie_bracket"] == ["0", "0"]

    @pytest.mark.asyncio
    async def test_bracket_rotation(self, tool, ctx):
        # X = -y ∂_x + x ∂_y, Y = ∂_x  →  [X, Y] 非零
        args = SymbolicMathInput(
            action="diffgeo_lie_derivative",
            symbols=["x", "y"],
            matrix=[["-y"], ["x"]],
            expression="1;0",
        )
        result = await tool.call(args, ctx)
        assert result.success
        bracket = result.data["lie_bracket"]
        # 至少一个分量非零
        assert any(b != "0" for b in bracket)


class TestDiffgeoConnection:
    """Levi-Civita 联络."""

    @pytest.mark.asyncio
    async def test_flat_connection_zero(self, tool, ctx):
        args = SymbolicMathInput(
            action="diffgeo_connection",
            symbols=["x", "y"],
            matrix=[["1", "0"], ["0", "1"]],
        )
        result = await tool.call(args, ctx)
        assert result.success, result.error
        # 第一类和第二类都应全零
        first = result.data["christoffel_first_kind"]
        second = result.data["christoffel_second_kind"]
        assert all(c == "0" for row in first for r in row for c in r)
        assert all(c == "0" for row in second for r in row for c in r)


class TestDeriveAlias:
    """derive action 应等价于 euler_lagrange."""

    @pytest.mark.asyncio
    async def test_derive_calls_euler_lagrange(self, tool, ctx):
        # L = (1/2) m u'^2 → derive 应返回 EL: -m u'' = 0
        args = SymbolicMathInput(
            action="derive",
            expression="m/2 * u'**2",
            symbols=["u", "x", "m"],
            variable="u",
        )
        result = await tool.call(args, ctx)
        assert result.success, result.error
        assert "euler_lagrange" in result.data
        el = result.data["euler_lagrange"]
        assert "m" in el
