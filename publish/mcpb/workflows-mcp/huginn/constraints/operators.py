"""Constraint operators.

A ``SafetyOperator`` enforces hard boundaries (permissions, sandboxing,
value ranges).  A ``QualityOperator`` checks physical reasonableness and
numerical quality.  Both operate on the same ``Constraint`` abstraction.
"""

from __future__ import annotations

from typing import Any

from huginn.constraints.reference import Constraint, ConstraintResult


class _BaseOperator:
    """Base class for safety/quality operators."""

    def __init__(self, constraints: list[Constraint] | None = None) -> None:
        self.constraints = constraints or []

    def add(self, constraint: Constraint) -> None:
        """Register a constraint."""
        self.constraints.append(constraint)

    def evaluate(
        self, data: dict[str, Any], scope: str | None = None
    ) -> list[ConstraintResult]:
        """Evaluate all constraints that match the optional scope."""
        selected = self.constraints
        if scope is not None:
            selected = [c for c in self.constraints if c.scope == scope]
        results = []
        for c in selected:
            result = c.evaluate(data)
            result.family = c.family
            result.severity = c.severity
            results.append(result)
        return results


class SafetyOperator(_BaseOperator):
    """Enforces safety-related constraints."""


class QualityOperator(_BaseOperator):
    """Enforces quality/physical correctness constraints."""
