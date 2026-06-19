"""Tests for the unified scientific computing framework."""

from __future__ import annotations

import sympy as sp

from huginn.unified import (
    derive_equations,
    harmonic_oscillator_md,
    heat_equation_fem,
    linear_elasticity_fem,
    one_d_kohn_sham_dft,
)
from huginn.unified.bridge import cauchy_born_elasticity, dft_potential_to_md
from huginn.unified.core import VariationalPrinciple


def test_harmonic_oscillator_derivation() -> None:
    problem = harmonic_oscillator_md()
    assert problem.principle == VariationalPrinciple.HAMILTONIAN
    result = derive_equations(problem)
    eqs = result["equations"]
    p, q = sp.symbols("p q")
    assert sp.simplify(eqs["dq_dt"] - p) == 0
    assert sp.simplify(eqs["dp_dt"] + q) == 0


def test_heat_equation_derivation() -> None:
    problem = heat_equation_fem(k=1.0, f=0.0)
    result = derive_equations(problem)
    eq = result["equations"]["temperature"]
    # Should be -k T'' = 0 for f=0, k=1
    x = sp.Symbol("x")
    T = sp.Function("T")
    assert sp.simplify(eq.lhs + sp.diff(T(x), x, 2)) == 0
    assert eq.rhs == 0


def test_linear_elasticity_derivation() -> None:
    problem = linear_elasticity_fem(E=2.0, f=0.0)
    result = derive_equations(problem)
    eq = result["equations"]["displacement"]
    x = sp.Symbol("x")
    u = sp.Function("u")
    assert sp.simplify(eq.lhs + 2.0 * sp.diff(u(x), x, 2)) == 0
    assert eq.rhs == 0


def test_kohn_sham_dft_derivation() -> None:
    problem = one_d_kohn_sham_dft()
    result = derive_equations(problem)
    eqs = result["equations"]
    assert "kohn_sham_equations" in eqs
    assert "density_from_orbitals" in eqs
    assert "effective_potential" in eqs


def test_dft_to_md_bridge() -> None:
    dft_problem = one_d_kohn_sham_dft()
    result = dft_potential_to_md(dft_problem)
    assert "potential" in result
    assert "morse" in result["potential"].name.lower() or "fit" in result[
        "potential"
    ].parameters.get("form", "")


def test_cauchy_born_bridge() -> None:
    import sympy as sp

    from huginn.unified.bridge import ConstitutiveModel

    r = sp.Symbol("r")
    potential = ConstitutiveModel(
        name="harmonic",
        expression=sp.Rational(1, 2) * sp.Symbol("k") * (r - sp.Symbol("r0")) ** 2,
    )
    result = cauchy_born_elasticity(potential)
    assert "elastic_modulus" in result
