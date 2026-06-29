"""Reference domain for constraints.

Defines the meta-constraint space: safety rules and quality invariants that
are independent of any specific task.  A ``Constraint`` is an abstract check
that can be mapped into concrete tool inputs / outputs by the adapter.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class ConstraintResult:
    """Outcome of evaluating a single constraint."""

    name: str
    passed: bool
    value: Any
    expected: str
    tolerance: float
    message: str
    severity: str = "warn"
    family: str = "quality"
    # 数值奖励通道: None 时由聚合层从 passed 推导 (passed→1.0, failed→0.0)
    score: float | None = None


@dataclass
class Constraint:
    """An abstract constraint in the reference domain.

    Attributes:
        name: Human-readable identifier.
        scope: Domain this constraint applies to, e.g. ``"dft"``, ``"md"``.
        family: ``"safety"`` or ``"quality"``.
        severity: ``"warn"``, ``"block"`` or ``"retry"``.
        check: Callable that receives result data and returns a
            ``ConstraintResult``.
    """

    name: str
    scope: str
    family: str  # "safety" | "quality"
    severity: str  # "warn" | "block" | "retry"
    check: Callable[[dict[str, Any]], ConstraintResult]

    def evaluate(self, data: dict[str, Any]) -> ConstraintResult:
        """Evaluate this constraint against concrete result data."""
        return self.check(data)
