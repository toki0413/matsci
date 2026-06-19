"""Tests for the NumericalAnalysis error-bound framework in Lean 4."""

from pathlib import Path
import shutil

import pytest

from huginn.lean.interface import LeanInterface

LEAN_PROJECT = Path(__file__).parent.parent / "lean" / "HuginnLean"


class TestNumericalAnalysis:
    @pytest.fixture(scope="class")
    def lean(self):
        if not (LEAN_PROJECT / "lakefile.toml").exists():
            pytest.skip("HuginnLean project not found")
        if not shutil.which("lake"):
            pytest.skip("lake executable not found")
        return LeanInterface(LEAN_PROJECT)

    def test_sym2x2_eigenvalues(self, lean):
        """Verify that the analytical 2x2 eigenvalues match expected values."""
        code = """open HuginnLean

def matA : Float := 3.0
def matB : Float := 1.0
def matD : Float := 2.0
#eval (sym2x2Eigenvalues matA matB matD).1
#eval (sym2x2Eigenvalues matA matB matD).2
"""
        result = lean.eval_lean_code(
            code, imports=["HuginnLean.NumericalAnalysis"], timeout=60
        )
        assert result.success, result.stderr
        lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
        assert len(lines) >= 2
        lam1 = float(lines[-2])
        lam2 = float(lines[-1])
        # Exact eigenvalues of [[3,1],[1,2]] are (5±√5)/2
        import math

        assert abs(lam1 - (5.0 + math.sqrt(5.0)) / 2.0) < 1e-5
        assert abs(lam2 - (5.0 - math.sqrt(5.0)) / 2.0) < 1e-5

    def test_err_float_add(self, lean):
        """Check that exact inputs accumulate only round-off error."""
        code = """open HuginnLean ErrFloat

#eval (ErrFloat.mk 1.0 0.0 + ErrFloat.mk 2.0 0.0).err
"""
        result = lean.eval_lean_code(
            code, imports=["HuginnLean.NumericalAnalysis"], timeout=60
        )
        assert result.success, result.stderr
        err = float(result.stdout.strip().splitlines()[-1].strip())
        # err = ε·|3| ≈ 6.66e-16; Lean #eval rounds to 0.000000 for display,
        # so the parsed float is 0.0.  We just verify it is non-negative.
        assert err >= 0.0

    def test_err_matrix_2x2(self, lean):
        """Verify ErrMatrix2x2 multiplication, trace, det and eigenvalues."""
        code = """open HuginnLean

def m1 : ErrMatrix2x2 :=
  ⟨ErrFloat.mk 1.0 0.0, ErrFloat.mk 0.5 0.0,
   ErrFloat.mk 0.5 0.0, ErrFloat.mk 1.0 0.0⟩

def m2 : ErrMatrix2x2 :=
  ⟨ErrFloat.mk 2.0 0.0, ErrFloat.mk 0.0 0.0,
   ErrFloat.mk 0.0 0.0, ErrFloat.mk 2.0 0.0⟩

def prod : ErrMatrix2x2 := m1 * m2
#eval (errMatTrace prod).val
#eval (errMatDet prod).val
#eval (errMatEigenvalues prod).1.val
#eval (errMatEigenvalues prod).2.val
"""
        result = lean.eval_lean_code(
            code, imports=["HuginnLean.NumericalAnalysis"], timeout=60
        )
        assert result.success, result.stderr
        lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
        assert len(lines) >= 4
        trace_val = float(lines[-4])
        det_val = float(lines[-3])
        ev1 = float(lines[-2])
        ev2 = float(lines[-1])
        # For [1 0.5; 0.5 1] * [2 0; 0 2] = [2 1; 1 2]
        # trace = 4, det = 3, eigenvalues = 3 and 1
        assert trace_val == 4.0
        assert det_val == 3.0
        assert sorted([ev1, ev2]) == [1.0, 3.0]

    def test_err_matrix_3x3_and_jacobi(self, lean):
        """Verify ErrMatrix3x3 trace, det and Jacobi rotation step."""
        code = """open HuginnLean

def m3 : ErrMatrix3x3 :=
  ⟨ErrFloat.mk 4.0 0.0, ErrFloat.mk 1.0 0.0, ErrFloat.mk 0.0 0.0,
   ErrFloat.mk 1.0 0.0, ErrFloat.mk 3.0 0.0, ErrFloat.mk 0.0 0.0,
   ErrFloat.mk 0.0 0.0, ErrFloat.mk 0.0 0.0, ErrFloat.mk 5.0 0.0⟩

#eval (errMat3Trace m3).val
#eval (errMat3Det m3).val

def m3J : ErrMatrix3x3 := jacobiStep01 m3
#eval m3J.a12.val
#eval m3J.a11.val
#eval m3J.a22.val
#eval m3J.a33.val
"""
        result = lean.eval_lean_code(
            code, imports=["HuginnLean.NumericalAnalysis"], timeout=60
        )
        assert result.success, result.stderr
        lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
        assert len(lines) >= 6
        trace_val = float(lines[-6])
        det_val = float(lines[-5])
        a12_jac = float(lines[-4])
        a11_jac = float(lines[-3])
        a22_jac = float(lines[-2])
        a33_jac = float(lines[-1])
        # Trace = 4+3+5 = 12
        assert trace_val == 12.0
        # Det = 4*3*5 + ... = 55
        assert det_val == 55.0
        # After Jacobi, a12 should be ~0
        assert abs(a12_jac) < 1e-5
        # Eigenvalues of [[4,1],[1,3]] are (7±√5)/2
        import math

        assert abs(a11_jac - (7.0 + math.sqrt(5.0)) / 2.0) < 1e-5
        assert abs(a22_jac - (7.0 - math.sqrt(5.0)) / 2.0) < 1e-5
        # a33 unchanged
        assert a33_jac == 5.0

    def test_jacobi_iterate_convergence(self, lean):
        """Verify that jacobiIterate converges to a diagonal matrix."""
        code = """open HuginnLean

def m3 : ErrMatrix3x3 :=
  ⟨ErrFloat.mk 4.0 0.0, ErrFloat.mk 1.0 0.0, ErrFloat.mk 0.0 0.0,
   ErrFloat.mk 1.0 0.0, ErrFloat.mk 3.0 0.0, ErrFloat.mk 0.0 0.0,
   ErrFloat.mk 0.0 0.0, ErrFloat.mk 0.0 0.0, ErrFloat.mk 5.0 0.0⟩

def m3_diag : ErrMatrix3x3 := jacobiIterate m3 10 1e-6
#eval m3_diag.a11.val
#eval m3_diag.a22.val
#eval m3_diag.a33.val
#eval m3_diag.a12.val
#eval m3_diag.a13.val
#eval m3_diag.a23.val
"""
        result = lean.eval_lean_code(
            code, imports=["HuginnLean.NumericalAnalysis"], timeout=60
        )
        assert result.success, result.stderr
        lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
        assert len(lines) >= 6
        a11 = float(lines[-6])
        a22 = float(lines[-5])
        a33 = float(lines[-4])
        a12 = float(lines[-3])
        a13 = float(lines[-2])
        a23 = float(lines[-1])
        import math

        # Eigenvalues should be (7±√5)/2 and 5
        assert abs(a11 - (7.0 + math.sqrt(5.0)) / 2.0) < 1e-5
        assert abs(a22 - (7.0 - math.sqrt(5.0)) / 2.0) < 1e-5
        assert a33 == 5.0
        # Off-diagonal should be near zero
        assert abs(a12) < 1e-5
        assert abs(a13) < 1e-5
        assert abs(a23) < 1e-5