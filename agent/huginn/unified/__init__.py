"""Unified scientific computing framework for Huginn.

Provides a common mathematical language for DFT, MD, FEM, CFD and other
computational methods based on first-principles and mathematical structuralism.
"""

from huginn.unified.core import (
    Domain,
    DomainType,
    EnergyFunctional,
    Field,
    FieldKind,
    UnifiedProblem,
    VariationalPrinciple,
)
from huginn.unified.derive import derive_equations
from huginn.unified.discretize import discretize
from huginn.unified.models import (
    harmonic_oscillator_md,
    heat_equation_2d,
    heat_equation_fem,
    linear_elasticity_fem,
    one_d_kohn_sham_dft,
)
from huginn.unified.solve import solve
from huginn.unified.visualize import plot_solution, solve_and_plot

__all__ = [
    "Domain",
    "DomainType",
    "EnergyFunctional",
    "Field",
    "FieldKind",
    "UnifiedProblem",
    "VariationalPrinciple",
    "derive_equations",
    "discretize",
    "solve",
    "plot_solution",
    "solve_and_plot",
    "harmonic_oscillator_md",
    "heat_equation_2d",
    "heat_equation_fem",
    "linear_elasticity_fem",
    "one_d_kohn_sham_dft",
]
