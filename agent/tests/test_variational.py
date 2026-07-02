"""Tests for SymbolicMathTool variational actions.

Covers Euler-Lagrange equation, functional derivative, isoperimetric
(constrained variational), and Noether theorem.
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


class TestEulerLagrange:
    """EL equation: ∂L/∂u - d/dx(∂L/∂u') = 0."""

    @pytest.mark.asyncio
    async def test_free_particle(self, tool, ctx):
        # L = (1/2) m u'^2  →  EL: -m u'' = 0  ⇒  u'' = 0
        args = SymbolicMathInput(
            action="euler_lagrange",
            expression="m/2 * u'**2",
            symbols=["u", "x", "m"],
            variable="u",
        )
        result = await tool.call(args, ctx)
        assert result.success, result.error
        # EL should contain m * u'' (second derivative of u wrt x)
        el = result.data["euler_lagrange"]
        assert "m" in el
        assert "Derivative" in el or "u''" in el

    @pytest.mark.asyncio
    async def test_harmonic_oscillator(self, tool, ctx):
        # L = (1/2) m u'^2 - (1/2) k u^2  →  EL: -k u - m u'' = 0
        args = SymbolicMathInput(
            action="euler_lagrange",
            expression="m/2 * u'**2 - k/2 * u**2",
            symbols=["u", "x", "m", "k"],
            variable="u",
        )
        result = await tool.call(args, ctx)
        assert result.success
        el = result.data["euler_lagrange"]
        assert "u''" in el or "Derivative" in el
        assert "k" in el

    @pytest.mark.asyncio
    async def test_multivar_laplace(self, tool, ctx):
        # L = (1/2) (u_x^2 + u_y^2);  multivar EL → -Δu = 0
        args = SymbolicMathInput(
            action="euler_lagrange",
            target="multivar",
            expression="1/2 * (u_x**2 + u_y**2)",
            symbols=["x", "y", "u"],
            variable="u",
        )
        result = await tool.call(args, ctx)
        assert result.success, result.error
        assert result.data["spatial_vars"] == ["x", "y"]
        el = result.data["euler_lagrange"]
        assert "Derivative" in el or "0" in el


class TestFunctionalDerivative:
    """δF/δu = EL left-hand side."""

    @pytest.mark.asyncio
    async def test_quadratic_functional(self, tool, ctx):
        # F[u] = ∫ (u'^2 + u^2) dx  →  δF/δu = 2u - 2u''
        args = SymbolicMathInput(
            action="functional_derivative",
            expression="u'**2 + u**2",
            symbols=["u", "x"],
            variable="u",
        )
        result = await tool.call(args, ctx)
        assert result.success, result.error
        assert "functional_derivative" in result.data
        fd = result.data["functional_derivative"]
        assert "Derivative" in fd or "u''" in fd


class TestIsoperimetric:
    """Constrained variational problem via Lagrange multiplier."""

    @pytest.mark.asyncio
    async def test_catenary_like(self, tool, ctx):
        # 极值化弧长 ∫ sqrt(1 + u'^2) dx 在 ∫ u dx = const 约束下
        args = SymbolicMathInput(
            action="isoperimetric",
            expression="sqrt(1 + u'**2);u",
            symbols=["u", "x"],
            variable="u",
        )
        result = await tool.call(args, ctx)
        assert result.success, result.error
        assert "lambda" in result.data["augmented_lagrangian"]
        assert "sqrt" in result.data["functional_F"]
        el = result.data["euler_lagrange"]
        assert "Derivative" in el or "lambda" in el

    @pytest.mark.asyncio
    async def test_bad_format(self, tool, ctx):
        args = SymbolicMathInput(
            action="isoperimetric",
            expression="u",
            symbols=["u", "x"],
            variable="u",
        )
        result = await tool.call(args, ctx)
        assert not result.success
        assert "2 semicolon" in result.error


class TestNoether:
    """Noether: symmetry → conserved current."""

    @pytest.mark.asyncio
    async def test_translation_symmetry(self, tool, ctx):
        # L = u'^2 不显含 u → 平移对称 η=1, J = 2 u'
        args = SymbolicMathInput(
            action="noether",
            target="translation",
            expression="u'**2",
            symbols=["u", "x"],
            variable="u",
        )
        result = await tool.call(args, ctx)
        assert result.success, result.error
        assert result.data["eta"] == "1"
        J = result.data["conserved_current"]
        assert "Derivative(u(x), x)" in J or "u'" in J

    @pytest.mark.asyncio
    async def test_scaling_symmetry(self, tool, ctx):
        # L = u u' (scaling invariant-ish);  η = u, J = u * u'
        args = SymbolicMathInput(
            action="noether",
            target="scaling",
            expression="u * u'",
            symbols=["u", "x"],
            variable="u",
        )
        result = await tool.call(args, ctx)
        assert result.success
        assert result.data["symmetry"] == "u → (1 + ε) u"
        J = result.data["conserved_current"]
        assert "u(x)" in J

    @pytest.mark.asyncio
    async def test_custom_eta(self, tool, ctx):
        # L = u'^2, custom η = x  →  J = 2 x u'
        args = SymbolicMathInput(
            action="noether",
            target="custom",
            sub_action="x",
            expression="u'**2",
            symbols=["u", "x"],
            variable="u",
        )
        result = await tool.call(args, ctx)
        assert result.success
        assert result.data["eta"] == "x"
        J = result.data["conserved_current"]
        assert "x" in J
