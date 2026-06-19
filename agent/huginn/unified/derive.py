"""Symbolic derivation of governing equations from unified problem descriptions."""

from __future__ import annotations

from typing import Any

import sympy as sp

from huginn.unified.core import (
    EnergyFunctional,
    FieldKind,
    UnifiedProblem,
    VariationalPrinciple,
)


def _euler_lagrange(
    energy: sp.Expr, field: sp.Symbol, coords: list[sp.Symbol]
) -> sp.Expr:
    """Compute the Euler-Lagrange expression ∂E/∂φ - ∇·∂E/∂(∇φ).

    This assumes the energy density depends on φ and its first derivatives.
    """
    dE_dphi = sp.diff(energy, field)
    divergence_terms = []
    for x in coords:
        d_dx = sp.diff(energy, sp.diff(field, x))
        divergence_terms.append(sp.diff(d_dx, x))
    return sp.simplify(dE_dphi - sum(divergence_terms))


def _derive_hamiltonian(energy: EnergyFunctional) -> dict[str, sp.Expr]:
    """Derive Hamilton's equations from H(q, p)."""
    q, p = energy.variables
    return {
        "dq_dt": sp.diff(energy.expression, p),
        "dp_dt": -sp.diff(energy.expression, q),
    }


def _derive_self_consistent(problem: UnifiedProblem) -> dict[str, Any]:
    """Return the symbolic structure of a Kohn-Sham-style self-consistent problem."""
    density_field = next(
        (f for f in problem.fields.values() if "density" in f.name or f.name == "n"),
        None,
    )
    if density_field is None:
        density_field = next(iter(problem.fields.values()))
    n = density_field.expr()
    coords = problem.domain.coordinates if problem.domain else [sp.Symbol("x")]
    x = coords[0]
    psi = sp.Function("psi")(x)
    i = sp.Idx("i")
    N = sp.Symbol("N")
    v_eff = sp.Symbol("V_eff")
    return {
        "kohn_sham_equations": sp.Eq(
            sp.Function("H_KS")(n) * psi,
            v_eff * psi,
        ),
        "density_from_orbitals": sp.Eq(n, sp.Sum(sp.Abs(psi) ** 2, (i, 1, N))),
        "effective_potential": sp.Eq(
            v_eff,
            (
                problem.constitutive.expression
                if problem.constitutive
                else sp.Symbol("δE/δn")
            ),
        ),
    }


def derive_equations(problem: UnifiedProblem) -> dict[str, Any]:
    """Derive governing equations for a unified problem.

    Returns a dictionary mapping field names to their symbolic equations.
    """
    coords = problem.domain.coordinates if problem.domain else []
    principle = problem.principle

    if principle in {
        VariationalPrinciple.STATIONARY,
        VariationalPrinciple.MINIMUM,
        VariationalPrinciple.MAXIMUM,
    }:
        if problem.energy is None:
            raise ValueError("Energy functional required for variational principles")
        equations: dict[str, Any] = {}
        for field_name, field in problem.fields.items():
            primary = field.expr()
            if isinstance(primary, sp.Matrix):
                eqs = []
                for sym in primary:
                    eqs.append(
                        sp.Eq(
                            _euler_lagrange(problem.energy.expression, sym, coords), 0
                        )
                    )
                equations[field_name] = eqs
            else:
                equations[field_name] = sp.Eq(
                    _euler_lagrange(problem.energy.expression, primary, coords), 0
                )
        return {"principle": principle.value, "equations": equations}

    if principle == VariationalPrinciple.HAMILTONIAN:
        if problem.energy is None:
            raise ValueError("Hamiltonian required")
        return {
            "principle": principle.value,
            "equations": _derive_hamiltonian(problem.energy),
        }

    if principle == VariationalPrinciple.SELF_CONSISTENT:
        return {
            "principle": principle.value,
            "equations": _derive_self_consistent(problem),
        }

    if principle in {
        VariationalPrinciple.CONSERVATION,
        VariationalPrinciple.DISSIPATIVE,
    }:
        return {
            "principle": principle.value,
            "equations": {
                name: sp.Eq(
                    sp.diff(field.expr(), sp.Symbol("t")), sp.Symbol(f"source_{name}")
                )
                for name, field in problem.fields.items()
                if field.kind != FieldKind.PHASE_SPACE
            },
        }

    raise ValueError(f"Unsupported principle: {principle}")
