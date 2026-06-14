"""Unit tests for Lean 4 integration."""

import pytest
import sympy as sp
from pathlib import Path

from huginn.lean.interface import LeanInterface
from huginn.lean.sympy_to_lean import SymPyToLean


LEAN_PROJECT = Path(__file__).parent.parent / "lean" / "HuginnLean"


class TestLeanInterface:
    @pytest.fixture(scope="class")
    def lean(self):
        if not (LEAN_PROJECT / "lakefile.toml").exists():
            pytest.skip("HuginnLean project not found")
        return LeanInterface(LEAN_PROJECT)

    def test_build(self, lean):
        result = lean.build()
        assert result.success, f"Build failed: {result.stderr}"

    def test_verify_existing_theorem(self, lean):
        result = lean.verify_theorem("cauchy_stress_symmetry_iff_angular_momentum")
        assert result.success

    def test_run_lean_code_snippet(self, lean):
        code = "theorem foo : 1 + 1 = 2 := by rfl"
        result = lean.run_lean_code(code)
        assert result.success, f"Code snippet failed: {result.stderr}"

    def test_run_lean_code_fail(self, lean):
        code = "theorem bar : 1 + 1 = 3 := by rfl"
        result = lean.run_lean_code(code)
        assert not result.success


class TestSymPyToLean:
    @pytest.fixture
    def translator(self):
        return SymPyToLean()

    def test_integer(self, translator):
        assert translator.translate(sp.Integer(42)) == "42"

    def test_rational(self, translator):
        assert translator.translate(sp.Rational(3, 4)) == "(3 / 4)"

    def test_symbol(self, translator):
        x = sp.Symbol("x")
        assert translator.translate(x) == "x"

    def test_add_mul(self, translator):
        x, y = sp.symbols("x y")
        expr = x + 2 * y
        assert translator.translate(expr) == "x + 2 * y"

    def test_pow(self, translator):
        x = sp.Symbol("x")
        assert translator.translate(x**2) == "x ^ 2"

    def test_sin_cos(self, translator):
        x = sp.Symbol("x")
        expr = sp.sin(x)**2 + sp.cos(x)**2
        result = translator.translate(expr)
        assert "Real.sin" in result
        assert "Real.cos" in result

    def test_diff(self, translator):
        x = sp.Symbol("x")
        f = sp.Function("f")
        expr = sp.Derivative(f(x), x)
        result = translator.translate(expr)
        assert "deriv" in result

    def test_theorem_skeleton(self, translator):
        x = sp.Symbol("x")
        skeleton = translator.theorem_statement(
            "my_theorem",
            {"h1": x > 0},
            x + 1 > x,
        )
        assert "theorem my_theorem :" in skeleton
        assert "sorry" in skeleton


class TestLeanToolAutoVerify:
    @pytest.fixture(scope="class")
    def lean_tool(self):
        if not (LEAN_PROJECT / "lakefile.toml").exists():
            pytest.skip("HuginnLean project not found")
        from huginn.tools.lean_tool import LeanTool, LeanToolInput
        from huginn.types import ToolContext
        tool = LeanTool()
        return tool, LeanToolInput, ToolContext(session_id="test", workspace=".")

    @pytest.mark.asyncio
    async def test_auto_verify_constitutive(self, lean_tool):
        tool, Input, ctx = lean_tool
        result = await tool.call(
            Input(action="auto_verify", auto_verify_action="constitutive",
                  symbolic_result={"pressure": "-B0*(V0/V)**BP"},
                  symbols=["B0", "V0", "V", "BP"]),
            ctx,
        )
        assert result.success, result.error

    @pytest.mark.asyncio
    async def test_auto_verify_derivative(self, lean_tool):
        tool, Input, ctx = lean_tool
        result = await tool.call(
            Input(action="auto_verify", auto_verify_action="derivative",
                  original_expression="x**3 + 2*x**2",
                  variable="x",
                  expected_expression="3*x**2 + 4*x",
                  test_points={"x": 2.0}),
            ctx,
        )
        assert result.success, result.error

    @pytest.mark.asyncio
    async def test_auto_verify_eigenvalue(self, lean_tool):
        tool, Input, ctx = lean_tool
        result = await tool.call(
            Input(action="auto_verify", auto_verify_action="eigenvalue",
                  symbolic_result={
                      "eigenvalues": [{"value": "a + b"}, {"value": "a - b"}],
                      "trace": "2*a",
                  },
                  symbols=["a", "b"]),
            ctx,
        )
        assert result.success, result.error


    @pytest.mark.asyncio
    async def test_auto_verify_unified(self, lean_tool):
        tool, Input, ctx = lean_tool
        result = await tool.call(
            Input(
                action="auto_verify",
                auto_verify_action="unified",
                symbolic_result={
                    "model": "harmonic_oscillator_md",
                    "principle": "hamiltonian",
                    "energy_expression": "0.5*p**2 + 0.5*q**2",
                    "equations": {
                        "dq_dt": "1.0*p",
                        "dp_dt": "-1.0*q",
                    },
                },
                symbols=["p", "q"],
            ),
            ctx,
        )
        assert result.success, result.error
