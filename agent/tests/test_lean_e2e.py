"""End-to-end tests: SymPy computation → Lean 4 verification pipeline.

This tests the full bridge from Phase 1 (symbolic/numerical computation)
to Phase 2 (formal verification).
"""

import asyncio
import pytest
from pathlib import Path

from huginn.lean.interface import LeanInterface
from huginn.tools.symbolic_math_tool import SymbolicMathTool, SymbolicMathInput
from huginn.tools.lean_tool import LeanTool, LeanToolInput
from huginn.types import ToolContext


LEAN_PROJECT = Path(__file__).parent.parent / "lean" / "HuginnLean"
CTX = ToolContext(session_id="e2e", workspace=".")


class TestSymPyToLeanVerification:
    """Verify that numerical results computed in Python/SymPy can be
checked by compiling corresponding Lean 4 code."""

    @pytest.fixture(scope="class")
    def lean(self):
        if not (LEAN_PROJECT / "lakefile.toml").exists():
            pytest.skip("HuginnLean project not found")
        return LeanInterface(LEAN_PROJECT)

    def test_born_stability_computation_verification(self, lean):
        """Compute Born stability in Python, generate Lean 4 check, compile."""
        C11, C12, C44 = 230.0, 135.0, 117.0
        python_stable = C11 > abs(C12) and C11 + 2 * C12 > 0 and C44 > 0
        assert python_stable is True

        lean_code = f"""open HuginnLean

def testMaterial : CubicElastic := ⟨{C11}, {C12}, {C44}⟩
#eval cubicBornStable testMaterial
"""
        result = lean.eval_lean_code(lean_code, imports=["HuginnLean.Elasticity"])
        assert result.success, f"Lean verification failed: {result.stderr}"
        assert "true" in result.stdout, f"Expected 'true' in stdout, got: {result.stdout}"

    def test_hill_bounds_computation_verification(self, lean):
        """Verify Hill bounds hold for a known stable material."""
        C11, C12, C44 = 230.0, 135.0, 117.0

        lean_code = f"""open HuginnLean

def testMaterial : CubicElastic := ⟨{C11}, {C12}, {C44}⟩
#eval cubicHillBoundsHold testMaterial
"""
        result = lean.eval_lean_code(lean_code, imports=["HuginnLean.Elasticity"])
        assert result.success, f"Lean verification failed: {result.stderr}"
        assert "true" in result.stdout

    def test_unstable_material_fails_born(self, lean):
        """An unstable material should fail the Born check in Lean."""
        C11, C12, C44 = 100.0, 150.0, 50.0

        lean_code = f"""open HuginnLean

def badMaterial : CubicElastic := ⟨{C11}, {C12}, {C44}⟩
#eval cubicBornStable badMaterial
"""
        result = lean.eval_lean_code(lean_code, imports=["HuginnLean.Elasticity"])
        assert result.success, f"Lean verification failed: {result.stderr}"
        assert "false" in result.stdout

    def test_moduli_roundtrip(self, lean):
        """Python and Lean compute the same bulk modulus for iron."""
        C11, C12, C44 = 230.0, 135.0, 117.0
        python_Kv = (C11 + 2 * C12) / 3.0

        lean_code = f"""open HuginnLean

def iron : CubicElastic := ⟨{C11}, {C12}, {C44}⟩
#eval cubicBulkModulusVoigt iron
"""
        result = lean.eval_lean_code(lean_code, imports=["HuginnLean.Elasticity"])
        assert result.success
        lean_value = float(result.stdout.strip().split()[-1])
        assert abs(lean_value - python_Kv) < 1e-5


class TestCrossGoalSymbolicToLean:
    """Full pipeline: SymbolicMathTool → LeanTool auto_verify."""

    @pytest.fixture
    def sym_tool(self):
        return SymbolicMathTool()

    @pytest.fixture
    def lean_tool(self):
        if not (LEAN_PROJECT / "lakefile.toml").exists():
            pytest.skip("HuginnLean project not found")
        return LeanTool()

    def test_derivative_auto_verify_via_symbolic_result(self, sym_tool, lean_tool):
        """SymPy differentiates x^2, Lean auto_verify checks it numerically."""
        # Phase 1: symbolic derivation
        sym_result = asyncio.run(sym_tool.call(
            SymbolicMathInput(
                action="differentiate",
                expression="x**2",
                symbols=["x"],
                variable="x",
            ),
            CTX,
        ))
        assert sym_result.success
        assert sym_result.data["result"] == "2*x"

        # Phase 2: auto-verify in Lean 4 using symbolic_result dict
        lean_result = asyncio.run(lean_tool.call(
            LeanToolInput(
                action="auto_verify",
                auto_verify_action="derivative",
                symbolic_result=sym_result.data,
            ),
            CTX,
        ))
        assert lean_result.success, f"Lean auto_verify failed: {lean_result.error}"
        assert lean_result.data["verified"] is True

    def test_expression_auto_verify(self, lean_tool):
        """AutoLeanPipeline can compile a simple SymPy expression directly."""
        result = asyncio.run(lean_tool.call(
            LeanToolInput(
                action="auto_verify",
                auto_verify_action="derivative",
                original_expression="x**3",
                variable="x",
                expected_expression="3*x**2",
                test_points={"x": 2.0},
            ),
            CTX,
        ))
        assert result.success, f"Lean verification failed: {result.error}"
        assert result.data["verified"] is True
