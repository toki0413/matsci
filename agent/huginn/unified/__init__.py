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
from huginn.unified.discretize import DiscretizationMetadata, discretize
from huginn.unified.models import (
    harmonic_oscillator_md,
    heat_equation_2d,
    heat_equation_fem,
    linear_elasticity_fem,
    one_d_kohn_sham_dft,
)
from huginn.unified.solve import solve

# visualize.py imports matplotlib at module level. matplotlib is an optional
# runtime dep — eagerly importing it here would make every `from huginn.unified
# import solve/derive_equations/discretize` fail when matplotlib isn't
# installed (e.g. CI matrix without plotting libs). Defer via PEP 562
# module-level __getattr__ so the public names stay available but only trigger
# the matplotlib import when actually used.
_LAZY_VISUALIZE = {"plot_solution", "solve_and_plot"}


def __getattr__(name: str):
    if name in _LAZY_VISUALIZE:
        from huginn.unified import visualize

        return getattr(visualize, name)
    raise AttributeError(f"module 'huginn.unified' has no attribute {name!r}")


__all__ = [
    "DiscretizationMetadata",
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
