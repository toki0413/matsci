"""Dimensional validation re-export from execution layer.

Thin shim so callers can `from huginn.validation.dimensional import DimensionalValidator`
without reaching into execution internals.
"""

from huginn.execution.dimensional_validator import (
    DimensionalCheckResult,
    DimensionalValidator,
    PhysicalQuantity,
    Unit,
    UnitRegistry,
    registry,
)

__all__ = [
    "DimensionalCheckResult",
    "DimensionalValidator",
    "PhysicalQuantity",
    "Unit",
    "UnitRegistry",
    "registry",
]
