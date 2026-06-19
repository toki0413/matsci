"""Tests for tensor calculus: SymbolicMathTool tensor_calculus action and Lean 4 TensorAlgebra."""

import asyncio
import math

import pytest

from huginn.tools.lean_tool import LeanTool, LeanToolInput
from huginn.tools.symbolic_math_tool import SymbolicMathInput, SymbolicMathTool
from huginn.types import ToolContext

CTX = ToolContext(session_id="tensor_test", workspace=".")


class TestTensorCalculusSymPy:
    """Unit tests for SymbolicMathTool tensor_calculus action."""

    @pytest.fixture
    def tool(self):
        return SymbolicMathTool()

    def test_invariants_stress(self, tool):
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(
                    action="tensor_calculus",
                    expression="invariants",
                    tensor_type="stress",
                    voigt_vector=[100.0, 50.0, 30.0, 10.0, 5.0, 2.0],
                ),
                CTX,
            )
        )
        assert result.success, result.error
        inv = result.data["invariants"]
        # I1 = 100 + 50 + 30 = 180
        assert abs(inv["I1"] - 180.0) < 1e-9
        # I3 = det of full 3x3 matrix
        assert isinstance(inv["I3"], float)

    def test_deviatoric(self, tool):
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(
                    action="tensor_calculus",
                    expression="deviatoric",
                    tensor_type="stress",
                    voigt_vector=[30.0, 30.0, 30.0, 0.0, 0.0, 0.0],
                ),
                CTX,
            )
        )
        assert result.success, result.error
        dev = result.data["deviatoric_voigt"]
        # Hydrostatic stress => deviatoric is zero
        assert abs(dev[0]) < 1e-9
        assert abs(dev[1]) < 1e-9
        assert abs(dev[2]) < 1e-9
        assert abs(dev[3]) < 1e-9
        assert abs(dev[4]) < 1e-9
        assert abs(dev[5]) < 1e-9

    def test_principal_values(self, tool):
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(
                    action="tensor_calculus",
                    expression="principal",
                    tensor_type="stress",
                    voigt_vector=[3.0, 2.0, 1.0, 0.0, 0.0, 0.0],
                ),
                CTX,
            )
        )
        assert result.success, result.error
        pv = result.data["principal_values"]
        assert len(pv) == 3
        # Diagonal matrix => principal values are diagonal entries
        assert abs(pv[0] - 3.0) < 1e-9
        assert abs(pv[1] - 2.0) < 1e-9
        assert abs(pv[2] - 1.0) < 1e-9

    def test_von_mises(self, tool):
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(
                    action="tensor_calculus",
                    expression="von_mises",
                    tensor_type="stress",
                    voigt_vector=[100.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ),
                CTX,
            )
        )
        assert result.success, result.error
        vm = result.data["von_mises"]
        # Uniaxial tension σ11: von Mises = σ11
        assert abs(vm - 100.0) < 1e-6

    def test_rotate(self, tool):
        # 45° rotation about z-axis
        sqrt2_2 = math.sqrt(2) / 2
        R = [
            [sqrt2_2, -sqrt2_2, 0.0],
            [sqrt2_2, sqrt2_2, 0.0],
            [0.0, 0.0, 1.0],
        ]
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(
                    action="tensor_calculus",
                    expression="rotate",
                    tensor_type="stress",
                    voigt_vector=[100.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    rotation_matrix=R,
                ),
                CTX,
            )
        )
        assert result.success, result.error
        rot = result.data["rotated_voigt"]
        # After 45° rotation about z, σ11' = σ22' = 50
        assert abs(rot[0] - 50.0) < 1e-6
        assert abs(rot[1] - 50.0) < 1e-6

    def test_stiffness_invariants(self, tool):
        # Cubic stiffness (C11=100, C12=40, C44=30)
        # 21 components in upper-triangle order
        voigt_21 = [
            100.0,
            40.0,
            40.0,
            0.0,
            0.0,
            0.0,
            100.0,
            40.0,
            0.0,
            0.0,
            0.0,
            100.0,
            0.0,
            0.0,
            0.0,
            30.0,
            0.0,
            0.0,
            30.0,
            0.0,
            30.0,
        ]
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(
                    action="tensor_calculus",
                    expression="invariants",
                    tensor_type="stiffness",
                    voigt_vector=voigt_21,
                ),
                CTX,
            )
        )
        assert result.success, result.error
        ev = result.data["voigt_eigenvalues"]
        assert len(ev) == 6
        # Cubic stiffness eigenvalues: C11+2C12, C11-C12 (2×), C44 (3×)
        assert result.data["is_positive_definite"] is True
        assert any(abs(e - (100.0 + 2 * 40.0)) < 1e-3 for e in ev)
        assert any(abs(e - (100.0 - 40.0)) < 1e-3 for e in ev)
        assert any(abs(e - 30.0) < 1e-3 for e in ev)

    def test_stiffness_apply_strain(self, tool):
        # Isotropic stiffness: λ=50, μ=30 => C11=110, C12=50, C44=30
        # 21 components
        voigt_21 = [
            110.0,
            50.0,
            50.0,
            0.0,
            0.0,
            0.0,
            110.0,
            50.0,
            0.0,
            0.0,
            0.0,
            110.0,
            0.0,
            0.0,
            0.0,
            30.0,
            0.0,
            0.0,
            30.0,
            0.0,
            30.0,
        ]
        # Uniaxial strain [0.01, 0, 0, 0, 0, 0]
        strain = [[0.01, 0.0, 0.0, 0.0, 0.0, 0.0]]
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(
                    action="tensor_calculus",
                    expression="apply_to_strain",
                    tensor_type="stiffness",
                    voigt_vector=voigt_21,
                    rotation_matrix=strain,
                ),
                CTX,
            )
        )
        assert result.success, result.error
        sigma = result.data["stress_voigt"]
        # σ11 = C11*ε11 = 110*0.01 = 1.1
        assert abs(sigma[0] - 1.1) < 1e-9
        # σ22 = σ33 = C12*ε11 = 50*0.01 = 0.5
        assert abs(sigma[1] - 0.5) < 1e-9
        assert abs(sigma[2] - 0.5) < 1e-9


class TestTensorAlgebraLean:
    """Integration tests: Lean 4 TensorAlgebra module."""

    @pytest.fixture(scope="class")
    def lean(self):
        from pathlib import Path

        from huginn.lean.interface import LeanInterface

        project = Path(__file__).parent.parent / "lean" / "HuginnLean"
        if not (project / "lakefile.toml").exists():
            pytest.skip("HuginnLean project not found")
        return LeanInterface(project)

    def test_isotropic_stiffness_applied(self, lean):
        """Apply isotropic stiffness to pure shear strain."""
        code = """open HuginnLean

-- λ = 50 GPa, μ = 30 GPa => C11 = 110, C12 = 50, C44 = 30
def C : FourthOrderTensor := isotropicElasticity 50.0 30.0

-- Pure shear strain ε12 = 0.01
def eps : SecondOrderTensor3 := ⟨0.0, 0.0, 0.0, 0.0, 0.0, 0.01⟩

def sigma : SecondOrderTensor3 := applyStiffness C eps

#eval sigma.s12
#eval sigma.s11
"""
        result = lean.eval_lean_code(
            code, imports=["HuginnLean.TensorAlgebra"], timeout=60
        )
        assert result.success, result.stderr
        lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
        assert len(lines) >= 2
        tau = float(lines[-2])  # σ12 = 2*C44*ε12 = 2*30*0.01 = 0.6
        s11 = float(lines[-1])  # σ11 = 0
        assert abs(tau - 0.6) < 1e-5
        assert abs(s11) < 1e-5

    def test_cubic_invariants(self, lean):
        """Compute invariants of a stress tensor."""
        code = """open HuginnLean

def sigma : SecondOrderTensor3 := ⟨100.0, 50.0, 30.0, 10.0, 5.0, 2.0⟩

#eval (invariantsSOT sigma).1
#eval (invariantsSOT sigma).2.1
#eval (invariantsSOT sigma).2.2
"""
        result = lean.eval_lean_code(
            code, imports=["HuginnLean.TensorAlgebra"], timeout=60
        )
        assert result.success, result.stderr
        lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
        assert len(lines) >= 3
        I1 = float(lines[-3])
        I2 = float(lines[-2])
        I3 = float(lines[-1])
        assert abs(I1 - 180.0) < 1e-5
        assert I2 > 0  # should be positive for this tensor
        assert isinstance(I3, float)

    def test_von_mises_hydrostatic(self, lean):
        """Hydrostatic stress should give zero von Mises."""
        code = """open HuginnLean

def sigma : SecondOrderTensor3 := ⟨30.0, 30.0, 30.0, 0.0, 0.0, 0.0⟩
#eval vonMisesStress sigma
"""
        result = lean.eval_lean_code(
            code, imports=["HuginnLean.TensorAlgebra"], timeout=60
        )
        assert result.success, result.stderr
        vm = float(result.stdout.strip().splitlines()[-1].strip())
        assert abs(vm) < 1e-5

    def test_deviatoric_trace_zero(self, lean):
        """Deviatoric tensor should have zero trace."""
        code = """open HuginnLean

def sigma : SecondOrderTensor3 := ⟨100.0, 50.0, 30.0, 10.0, 5.0, 2.0⟩
def s : SecondOrderTensor3 := deviatoricSOT sigma
#eval traceSOT s
"""
        result = lean.eval_lean_code(
            code, imports=["HuginnLean.TensorAlgebra"], timeout=60
        )
        assert result.success, result.stderr
        tr = float(result.stdout.strip().splitlines()[-1].strip())
        assert abs(tr) < 1e-5


class TestTensorCalculusCrossGoal:
    """End-to-end: SymbolicMathTool → LeanTool auto_verify."""

    @pytest.fixture
    def sym_tool(self):
        return SymbolicMathTool()

    @pytest.fixture
    def lean_tool(self):
        from pathlib import Path

        project = Path(__file__).parent.parent / "lean" / "HuginnLean"
        if not (project / "lakefile.toml").exists():
            pytest.skip("HuginnLean project not found")
        return LeanTool()

    def test_invariants_auto_verify(self, sym_tool, lean_tool):
        sym_result = asyncio.run(
            sym_tool.call(
                SymbolicMathInput(
                    action="tensor_calculus",
                    expression="invariants",
                    tensor_type="stress",
                    voigt_vector=[3.0, 2.0, 1.0, 0.0, 0.0, 0.0],
                ),
                CTX,
            )
        )
        assert sym_result.success

        lean_result = asyncio.run(
            lean_tool.call(
                LeanToolInput(
                    action="auto_verify",
                    auto_verify_action="tensor_calculus",
                    symbolic_result=sym_result.data,
                ),
                CTX,
            )
        )
        assert lean_result.success, f"Lean verification failed: {lean_result.error}"
        assert lean_result.data["verified"] is True
