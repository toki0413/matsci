"""Tests for SymbolicMathTool PDE actions: classify / separation / characteristics / discretize."""

from __future__ import annotations

import pytest

from huginn.tools.symbolic_math.tool import SymbolicMathInput, SymbolicMathTool
from huginn.types import ToolContext


@pytest.fixture
def tool() -> SymbolicMathTool:
    return SymbolicMathTool()


@pytest.fixture
def ctx() -> ToolContext:
    return ToolContext(session_id="test", workspace=".")


class TestPdeClassify:
    """判别式 B^2 - 4AC 给二阶 PDE 分类."""

    @pytest.mark.asyncio
    async def test_elliptic_laplace(self, tool, ctx):
        # A=1, B=0, C=1  →  B^2 - 4AC = -4 < 0  →  elliptic
        args = SymbolicMathInput(action="pde_classify", expression="1;0;1")
        result = await tool.call(args, ctx)
        assert result.success, result.error
        assert result.data["classification"] == "elliptic"
        assert result.data["discriminant_value"].real < 0
        assert "Laplace" in result.data["canonical_form"]

    @pytest.mark.asyncio
    async def test_hyperbolic_wave(self, tool, ctx):
        # A=1, B=0, C=-1  →  B^2 - 4AC = 4 > 0  →  hyperbolic
        args = SymbolicMathInput(action="pde_classify", expression="1;0;-1")
        result = await tool.call(args, ctx)
        assert result.success
        assert result.data["classification"] == "hyperbolic"
        assert result.data["discriminant_value"].real > 0

    @pytest.mark.asyncio
    async def test_parabolic_heat(self, tool, ctx):
        # A=0, B=0, C=1  →  B^2 - 4AC = 0  →  parabolic (heat equation u_t = u_xx)
        args = SymbolicMathInput(action="pde_classify", expression="0;0;1")
        result = await tool.call(args, ctx)
        assert result.success
        assert result.data["classification"] == "parabolic"

    @pytest.mark.asyncio
    async def test_bad_format(self, tool, ctx):
        args = SymbolicMathInput(action="pde_classify", expression="1;0")
        result = await tool.call(args, ctx)
        assert not result.success
        assert "3 semicolon" in result.error


class TestPdeSeparation:
    """分离变量法: heat / wave / laplace."""

    @pytest.mark.asyncio
    async def test_heat_separation(self, tool, ctx):
        args = SymbolicMathInput(
            action="pde_separation", target="heat", symbols=["k"]
        )
        result = await tool.call(args, ctx)
        assert result.success, result.error
        assert "X''" in result.data["spatial_ode"]
        assert "T'(t)" in result.data["temporal_ode"]
        assert "exp(-λ" in result.data["general_solution"]
        assert "Dirichlet" in result.data["eigenvalues"]

    @pytest.mark.asyncio
    async def test_wave_separation(self, tool, ctx):
        args = SymbolicMathInput(
            action="pde_separation", target="wave", symbols=["c"]
        )
        result = await tool.call(args, ctx)
        assert result.success
        assert "T''(t)" in result.data["temporal_ode"]
        assert "cos" in result.data["general_solution"]

    @pytest.mark.asyncio
    async def test_laplace_separation(self, tool, ctx):
        args = SymbolicMathInput(action="pde_separation", target="laplace")
        result = await tool.call(args, ctx)
        assert result.success
        assert "Y''(y)" in result.data["temporal_ode"]
        assert "cosh" in result.data["general_solution"]


class TestPdeCharacteristics:
    """一阶 PDE 特征线法."""

    @pytest.mark.asyncio
    async def test_transport(self, tool, ctx):
        # u_t + c u_x = 0  →  u(x,t) = F(x - c t)
        args = SymbolicMathInput(
            action="pde_characteristics",
            target="transport",
            expression="c",
            symbols=["c"],
        )
        result = await tool.call(args, ctx)
        assert result.success, result.error
        assert "x -" in result.data["invariant"]
        assert "c t = const" in result.data["invariant"]
        assert "F(x -" in result.data["general_solution"]
        assert "t)" in result.data["general_solution"]

    @pytest.mark.asyncio
    async def test_first_order_linear(self, tool, ctx):
        # a u_x + b u_y = f  →  dx/ds=a, dy/ds=b, du/ds=f
        args = SymbolicMathInput(
            action="pde_characteristics",
            target="first_order_linear",
            expression="y;x;0",
            symbols=["x", "y"],
        )
        result = await tool.call(args, ctx)
        assert result.success
        assert len(result.data["characteristic_system"]) == 3
        assert "dx/ds" in result.data["characteristic_system"][0]


class TestPdeDiscretize:
    """有限差分 stencil 生成."""

    @pytest.mark.asyncio
    async def test_laplacian_2d(self, tool, ctx):
        args = SymbolicMathInput(
            action="pde_discretize", target="laplacian_2d"
        )
        result = await tool.call(args, ctx)
        assert result.success, result.error
        assert "5-point" in result.data["scheme"]
        assert "-4/h^2" in result.data["stencil_points"][0]
        assert "O(h^2)" in result.data["order"]

    @pytest.mark.asyncio
    async def test_laplacian_3d(self, tool, ctx):
        args = SymbolicMathInput(
            action="pde_discretize", target="laplacian_3d"
        )
        result = await tool.call(args, ctx)
        assert result.success
        assert "7-point" in result.data["scheme"]
        assert "-6/h^2" in result.data["stencil_points"][0]

    @pytest.mark.asyncio
    async def test_heat_ftcs_stability(self, tool, ctx):
        args = SymbolicMathInput(
            action="pde_discretize",
            target="heat_ftcs",
            symbols=["h", "dt", "alpha"],
        )
        result = await tool.call(args, ctx)
        assert result.success
        assert "r = α dt / h^2" in result.data["courant_number"]
        assert "<= 1/2" in result.data["stability"]

    @pytest.mark.asyncio
    async def test_wave_explicit_cfl(self, tool, ctx):
        args = SymbolicMathInput(
            action="pde_discretize",
            target="wave_explicit",
            symbols=["h", "dt", "c"],
        )
        result = await tool.call(args, ctx)
        assert result.success
        assert "CFL = c dt / h" in result.data["courant_number"]
        assert "<= 1" in result.data["stability"]
