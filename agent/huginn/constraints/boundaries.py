"""Boundary evolution for the real domain.

A ``BoundaryState`` captures the current operational envelope of the Agent:
which executables are allowed, how long a command may run, whether human
confirmation is required, etc.  ``BoundaryEvolution`` adjusts this envelope
based on constraint evaluation history.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from huginn.constraints.reference import ConstraintResult


@dataclass
class BoundaryState:
    """Current behavioral boundary of the Agent in a task context."""

    allowed_executables: set[str] = field(default_factory=set)
    max_timeout: float = 3600.0
    require_confirmation: bool = True
    max_retries: int = 3
    blocked_tools: set[str] = field(default_factory=set)
    blocked_scopes: set[str] = field(default_factory=set)

    def allows(self, executable: str) -> bool:
        if not self.allowed_executables:
            return True
        return executable in self.allowed_executables


class BoundaryEvolution:
    """Adjust boundaries based on constraint results over time."""

    def __init__(self, state: BoundaryState | None = None) -> None:
        self.state = state or BoundaryState()
        self._history: list[list[ConstraintResult]] = []

    def update(self, results: list[ConstraintResult]) -> BoundaryState:
        """Update boundary state given latest constraint results.

        Current rules (minimal):
        - Any ``block``-severity failure disables auto-approval.
        - Any safety failure reduces retry budget to 1.
        """
        self._history.append(results)
        for r in results:
            if r.severity == "block" and not r.passed:
                self.state.require_confirmation = True
                self.state.max_retries = 1
        return self.state
