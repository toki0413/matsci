"""Multiscale bridges between first-principles, atomistic and continuum descriptions."""

from __future__ import annotations

from typing import Any

import sympy as sp

from huginn.unified.core import ConstitutiveModel, UnifiedProblem


def dft_potential_to_md(dft_problem: UnifiedProblem) -> ConstitutiveModel:
    """Extract an effective atomistic potential from a DFT energy surface.

    In the simplest approximation, replace the electron-density functional by a
    pair potential V(r) fitted from the DFT total energy. This is the conceptual
    bridge from quantum mechanics to classical MD.
    """
    r = sp.Symbol("r", positive=True)
    # Morse-like effective potential; in practice this would be fitted to DFT data.
    V = sp.Symbol("D_e") * (1 - sp.exp(-sp.Symbol("a") * (r - sp.Symbol("r_e")))) ** 2
    return {
        "interpretation": "DFT total energy surface is projected onto an effective pair potential V(r) for classical MD",
        "potential": ConstitutiveModel(
            name="effective_pair_potential",
            expression=V,
            parameters={"origin": dft_problem.name, "form": "morse_fit"},
        ),
    }


def md_stress_to_continuum() -> dict[str, Any]:
    """Return the symbolic link between MD virial stress and Cauchy stress.

    The continuum Cauchy stress σ is the ensemble / volume average of the
    Irving-Kirkwood virial stress:

        σ = (1/Ω) Σ (m_i v_i⊗v_i + ½ Σ r_ij ⊗ f_ij)
    """
    Omega = sp.Symbol("Omega")
    i = sp.Idx("i")
    j = sp.Idx("j")
    m, v, f, r = sp.IndexedBase("m"), sp.IndexedBase("v"), sp.IndexedBase("f"), sp.IndexedBase("r")
    kinetic = m[i] * v[i] ** 2
    virial = sp.Rational(1, 2) * r[i, j] * f[i, j]
    sigma = (kinetic + virial) / Omega
    return {
        "cauchy_stress": sp.Eq(sp.Symbol("sigma"), sigma),
        "interpretation": "Continuum Cauchy stress = volume average of MD virial stress",
    }


def cauchy_born_elasticity(potential: ConstitutiveModel) -> dict[str, Any]:
    """Cauchy-Born rule: derive continuum elastic constants from an atomistic potential.

    For a simple pair potential V(r), the second derivative at equilibrium gives
    the elastic modulus: C = V''(r_0) / (2 r_0) in 1D.
    """
    r = sp.Symbol("r")
    r0 = sp.Symbol("r_0")
    V = potential.expression
    C = sp.diff(V, r, 2).subs(r, r0)
    return {
        "elastic_modulus": sp.Eq(sp.Symbol("C"), C),
        "interpretation": "Cauchy-Born: continuum modulus from atomistic potential curvature",
    }


BRIDGE_REGISTRY = {
    "dft_to_md": dft_potential_to_md,
    "md_to_stress": md_stress_to_continuum,
    "md_to_elasticity": cauchy_born_elasticity,
}


def list_bridges() -> list[str]:
    return list(BRIDGE_REGISTRY.keys())


def get_bridge(name: str):
    return BRIDGE_REGISTRY.get(name)
