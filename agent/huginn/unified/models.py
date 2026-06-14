"""Pre-built unified models for common computational methods."""

from __future__ import annotations

import sympy as sp

from huginn.unified.core import (
    ConstitutiveModel,
    Domain,
    EnergyFunctional,
    Field,
    FieldKind,
    UnifiedProblem,
    VariationalPrinciple,
)


def harmonic_oscillator_md(mass: float = 1.0, k: float = 1.0) -> UnifiedProblem:
    """1D harmonic oscillator as a Hamiltonian problem.

    Maps to MD: nuclei moving in a quadratic potential.
    """
    q, p = sp.symbols("q p")
    domain = Domain.particles(n=1, dim=1)
    energy = EnergyFunctional(
        name="harmonic_oscillator_hamiltonian",
        expression=p**2 / (2 * mass) + sp.Rational(1, 2) * k * q**2,
        variables=[q, p],
        parameters={"mass": mass, "k": k},
        description="H = p²/(2m) + ½ k q²",
    )
    return UnifiedProblem(
        name="harmonic_oscillator_md",
        description="The simplest MD-like system: a mass on a spring.",
        fields={
            "phase_space": Field(
                name="phase_space",
                kind=FieldKind.PHASE_SPACE,
                symbols=[q, p],
                domain=domain,
                description="Generalized coordinate q and momentum p",
            )
        },
        principle=VariationalPrinciple.HAMILTONIAN,
        energy=energy,
        domain=domain,
    )


def heat_equation_fem(k: float = 1.0, f: float = 0.0) -> UnifiedProblem:
    """Steady-state heat equation as a minimum energy problem.

    Maps to FEM: thermal conduction with source term.
    """
    x = sp.Symbol("x")
    T = sp.Function("T")
    T_sym = T(x)
    domain = Domain.continuum_1d(x, bounds=(0.0, 1.0))
    energy_density = sp.Rational(1, 2) * k * sp.diff(T_sym, x) ** 2 - f * T_sym
    energy = EnergyFunctional(
        name="thermal_energy",
        expression=energy_density,
        variables=[T_sym],
        parameters={"k": k, "f": f},
        description="Π(T) = ∫ (½ k |∇T|² - f T) dx",
    )
    return UnifiedProblem(
        name="heat_equation_fem",
        description="Steady heat conduction derived from a minimum principle.",
        fields={
            "temperature": Field(
                name="temperature",
                kind=FieldKind.SCALAR,
                symbols=[T_sym],
                domain=domain,
                units="K",
                description="Temperature field T(x)",
            )
        },
        principle=VariationalPrinciple.MINIMUM,
        energy=energy,
        domain=domain,
    )


def linear_elasticity_fem(
    E: float = 1.0,
    nu: float = 0.3,
    f: float = 0.0,
) -> UnifiedProblem:
    """1D linear elasticity as a minimum potential energy problem.

    Maps to FEM: bar under axial load.
    """
    x = sp.Symbol("x")
    u = sp.Function("u")
    u_sym = u(x)
    domain = Domain.continuum_1d(x, bounds=(0.0, 1.0))
    # 1D strain energy density: ½ E (du/dx)² - f u
    energy_density = sp.Rational(1, 2) * E * sp.diff(u_sym, x) ** 2 - f * u_sym
    energy = EnergyFunctional(
        name="elastic_potential",
        expression=energy_density,
        variables=[u_sym],
        parameters={"E": E, "nu": nu, "f": f},
        description="Π(u) = ∫ (½ E (du/dx)² - f u) dx",
    )
    constitutive = ConstitutiveModel(
        name="linear_elastic",
        expression=E * sp.diff(u_sym, x),
        parameters={"E": E, "nu": nu},
    )
    return UnifiedProblem(
        name="linear_elasticity_fem",
        description="Axially loaded elastic bar derived from minimum potential energy.",
        fields={
            "displacement": Field(
                name="displacement",
                kind=FieldKind.VECTOR,
                symbols=[u_sym],
                domain=domain,
                units="m",
                description="Displacement field u(x)",
            )
        },
        principle=VariationalPrinciple.MINIMUM,
        energy=energy,
        domain=domain,
        constitutive=constitutive,
    )


def one_d_kohn_sham_dft() -> UnifiedProblem:
    """1D Kohn-Sham DFT as a self-consistent energy problem.

    Maps to DFT: electron density minimizes the Hohenberg-Kohn functional.
    """
    x = sp.Symbol("x")
    n = sp.Function("n")
    n_sym = n(x)
    domain = Domain.continuum_1d(x, bounds=(0.0, 1.0))

    # Simplified 1D energy functional: Thomas-Fermi-like kinetic + external + Hartree + XC
    v_ext = sp.Symbol("v_ext")
    C_tf = sp.Rational(3, 10)  # placeholder coefficient
    energy_expr = (
        C_tf * n_sym ** (sp.Rational(5, 3))
        + v_ext * n_sym
        + sp.Rational(1, 2) * n_sym * sp.Symbol("V_H")
        - sp.Rational(3, 4) * (3 / sp.pi) ** (sp.Rational(1, 3)) * n_sym ** (sp.Rational(4, 3))
    )
    energy = EnergyFunctional(
        name="hohenberg_kohn_1d",
        expression=energy_expr,
        variables=[n_sym],
        parameters={"v_ext": v_ext},
        description="E[n] = T_s[n] + ∫ v_ext n + E_H[n] + E_xc[n]",
    )
    xc = ConstitutiveModel(
        name="lda_xc_1d",
        expression=-(3 / sp.pi) ** (sp.Rational(1, 3)) * n_sym ** (sp.Rational(1, 3)),
        parameters={},
    )
    return UnifiedProblem(
        name="one_d_kohn_sham_dft",
        description="1D Kohn-Sham DFT derived from a density functional.",
        fields={
            "density": Field(
                name="density",
                kind=FieldKind.SCALAR,
                symbols=[n_sym],
                domain=domain,
                units="1/length",
                description="Electron density n(x)",
            )
        },
        principle=VariationalPrinciple.SELF_CONSISTENT,
        energy=energy,
        domain=domain,
        constitutive=xc,
    )


MODEL_REGISTRY = {
    "harmonic_oscillator_md": harmonic_oscillator_md,
    "heat_equation_fem": heat_equation_fem,
    "linear_elasticity_fem": linear_elasticity_fem,
    "one_d_kohn_sham_dft": one_d_kohn_sham_dft,
}


def list_models() -> list[str]:
    return list(MODEL_REGISTRY.keys())


def get_model(name: str):
    return MODEL_REGISTRY.get(name)
