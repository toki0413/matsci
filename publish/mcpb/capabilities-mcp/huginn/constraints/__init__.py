"""Constraint system for Huginn.

Maps abstract safety/quality invariants (Reference Domain) into concrete
checks against tool outputs (Real Domain) via a task-specific adapter.
"""

from huginn.constraints.adapter import (
    ConstraintAdapter,
    build_default_library,
)
from huginn.constraints.boundaries import BoundaryEvolution, BoundaryState
from huginn.constraints.operators import QualityOperator, SafetyOperator
from huginn.constraints.reference import Constraint, ConstraintResult

__all__ = [
    "Constraint",
    "ConstraintResult",
    "SafetyOperator",
    "QualityOperator",
    "ConstraintAdapter",
    "BoundaryState",
    "BoundaryEvolution",
    "build_default_library",
]
