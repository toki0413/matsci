"""Tests for the AutoLeanPipeline: symbolic → Lean 4 auto-verification."""

import pytest
from pathlib import Path

from huginn.lean.auto_pipeline import AutoLeanPipeline, FloatSymPyToLean


LEAN_PROJECT = Path(__file__).parent.parent / "lean" / "HuginnLean"


class TestFloatSymPyToLean:
    def test_basic_arithmetic(self):
        import sympy as sp
        x, y = sp.symbols("x y")
        t = FloatSymPyToLean()
        assert t.translate(x + 2 * y) == "x + 2 * y"
        assert t.translate(x ** 3) == "x ^ 3"
        assert t.translate(x ** 2.5) == "Float.pow x 2.5"

    def test_trig_functions(self):
        import sympy as sp
        x = sp.Symbol("x")
        t = FloatSymPyToLean()
        out = t.translate(sp.sin(x) ** 2 + sp.cos(x) ** 2)
        assert "(Float.sin x) ^ 2" in out
        assert "(Float.cos x) ^ 2" in out


class TestAutoLeanPipeline:
    @pytest.fixture(scope="class")
    def pipe(self):
        if not (LEAN_PROJECT / "lakefile.toml").exists():
            pytest.skip("HuginnLean project not found")
        return AutoLeanPipeline(LEAN_PROJECT)

    def test_verify_simple_expression(self, pipe):
        result = pipe.verify_expression("x**2 + 3*x", name="polyTest", symbols=["x"])
        assert result.success, result.stderr

    def test_verify_expression_dict(self, pipe):
        result = pipe.verify_expression_dict({
            "bulkModulus": "(c11 + 2*c12) / 3",
            "shearModulus": "(c11 - c12 + 3*c44) / 5",
        }, symbols=["c11", "c12", "c44"])
        assert result.success, result.stderr

    def test_verify_constitutive(self, pipe):
        """Simulate SymbolicMathTool constitutive output."""
        symbolic_result = {
            "second_pk_stress": "2*C10*(I1 - 3)",
            "pressure": "-B0*(V0/V)**BP",
            "bulk_modulus": "B0*(V0/V)**BP",
        }
        result = pipe.verify_constitutive(symbolic_result, symbols=["C10", "I1", "B0", "V0", "V", "BP"])
        assert result.success, result.stderr

    def test_verify_derivative_numerical(self, pipe):
        """Verify that symbolic derivative matches finite difference in Lean."""
        result = pipe.verify_derivative(
            original="x**3 + 2*x**2",
            variable="x",
            expected="3*x**2 + 4*x",
            test_points={"x": 2.0},
        )
        assert result.success, result.stderr
        lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
        assert len(lines) >= 2
        fd = float(lines[-2])
        sym = float(lines[-1])
        assert abs(fd - sym) < 1e-3  # finite-difference vs symbolic should be close

    def test_verify_weak_form(self, pipe):
        """Verify weak-form terms auto-convert to Lean."""
        result = pipe.verify_weak_form({
            "weak_form_terms": {"diffusion": "diff(u, x)*diff(v, x)", "reaction": "c*u*v"},
            "boundary_terms": {"neumann": "n_x*diff(u, x)*v"},
        }, symbols=["u", "v", "x", "c", "n_x"])
        assert result.success, result.stderr

    def test_verify_fem_bar(self, pipe):
        """Verify FEM bar element matrix auto-converts to Lean."""
        result = pipe.verify_fem({
            "element_type": "bar",
            "element_matrix": [[100.0, -100.0], [-100.0, 100.0]],
            "is_symmetric": True,
            "size": 2,
        })
        assert result.success, result.stderr

    def test_verify_fem_poisson_tri(self, pipe):
        """Verify FEM Poisson triangle weak form auto-converts to Lean."""
        result = pipe.verify_fem({
            "weak_form_terms": {"diffusion": "diff(u, x)*diff(v, x) + diff(u, y)*diff(v, y)"},
            "bilinear_form": "diff(u, x)*diff(v, x) + diff(u, y)*diff(v, y)",
            "linear_functional": "f*v",
            "domain_dim": 2,
            "element_type": "heat_conduction",
        }, symbols=["u", "v", "x", "y", "f"])
        assert result.success, result.stderr

    def test_verify_eigenvalue(self, pipe):
        """Verify eigenvalue expressions auto-convert to Lean."""
        result = pipe.verify_eigenvalue({
            "eigenvalues": [{"value": "a + b"}, {"value": "a - b"}],
            "trace": "2*a",
            "determinant": "a**2 - b**2",
        }, symbols=["a", "b"])
        assert result.success, result.stderr

    def test_verify_tensor_ops(self, pipe):
        """Verify tensor invariant expressions auto-convert to Lean."""
        result = pipe.verify_tensor_ops({
            "invariants": {"I1": "x + y", "I2": "x*y", "I3": "x**2 + y**2"},
            "trace": "x + y",
            "determinant": "x*y - 1",
        }, symbols=["x", "y"])
        assert result.success, result.stderr

    def test_verify_solve(self, pipe):
        """Verify solution expressions auto-convert to Lean."""
        result = pipe.verify_solve({
            "solutions": [{"x": "2", "y": "3"}, {"x": "-2", "y": "-3"}],
        })
        assert result.success, result.stderr

    def test_verify_linear_algebra_lu(self, pipe):
        """Verify LU decomposition results auto-convert to Lean."""
        result = pipe.verify_linear_algebra({
            "L": [["1.0", "0.0"], ["0.25", "1.0"]],
            "U": [["4.0", "1.0"], ["0.0", "2.75"]],
            "size": 2,
        })
        assert result.success, result.stderr

    def test_verify_linear_algebra_solution(self, pipe):
        """Verify solution vector auto-converts to Lean."""
        result = pipe.verify_linear_algebra({
            "solution": ["0.1818", "0.5455"],
            "size": 2,
        })
        assert result.success, result.stderr

    def test_verify_dft_fermi_energy(self, pipe):
        """Verify DFT Fermi energy auto-converts to Lean."""
        result = pipe.verify_dft({
            "fermi_energy": 0.1974,
            "fermi_wavevector": 0.4646,
            "density": 0.05,
        })
        assert result.success, result.stderr

    def test_verify_dft_lda_xc(self, pipe):
        """Verify LDA XC energy density auto-converts to Lean."""
        result = pipe.verify_dft({
            "xc_energy_density": -0.3141,
            "exchange_energy_density": -0.2285,
            "correlation_energy_density": -0.0856,
            "density": 0.05,
        })
        assert result.success, result.stderr

    def test_verify_thermo_ideal_gas(self, pipe):
        """Verify ideal gas results auto-convert to Lean."""
        result = pipe.verify_thermodynamics({
            "pressure": 101325.0,
            "internal_energy": 3406.6,
            "temperature": 273.15,
            "volume": 0.022414,
        })
        assert result.success, result.stderr

    def test_verify_thermo_vdw(self, pipe):
        """Verify van der Waals results auto-convert to Lean."""
        result = pipe.verify_thermodynamics({
            "pressure": 100793.7,
            "critical_temperature": 304.2,
            "critical_pressure": 7384041.3,
        })
        assert result.success, result.stderr

    def test_verify_probability_normal(self, pipe):
        """Verify normal PDF/CDF results auto-convert to Lean."""
        result = pipe.verify_probability({
            "pdf": 0.398942,
            "cdf": 0.500000,
            "mu": 0.0,
            "sigma": 1.0,
        })
        assert result.success, result.stderr

    def test_verify_probability_bayesian(self, pipe):
        """Verify Bayesian update results auto-convert to Lean."""
        result = pipe.verify_probability({
            "posterior_mean": 1.95122,
            "posterior_variance": 0.02439,
            "prior_mean": 0.0,
            "prior_variance": 1.0,
        })
        assert result.success, result.stderr

    def test_verify_matrix_eigenvalues_placeholder(self, pipe):
        """Matrix literals require a HuginnLean matrix type; skip for now."""
        pytest.skip("Lean 4 matrix literal syntax not yet available in HuginnLean")
