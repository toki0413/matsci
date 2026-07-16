"""Progressive resource budget for the autoloop.

As the autoloop iterates, tighter tiers kick in so late iterations can't keep
launching expensive DFT workflows. Each tier restricts which plan modes are
allowed and caps how many rejections the budget will issue before degrading to
"allow everything" — that's the safety valve so a stuck LLM doesn't stall the
loop forever.

Default tiers (调松后, 材料任务多轮 DFT 不再被过早一刀切):
  iterations 1-10   open    any mode, no rejection cap
  iterations 11-30  medium  coder / explore only, max 30 rejections
  iterations 31-50 light   coder only, max 20 rejections

The engine consults this right after _plan returns and before the plan->execute
phase-gate. A rejected plan injects a hint into _speculator_hint so the next
iteration's prompt carries "mode X not allowed at tier Y, use Z instead".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PlanMode = Literal["coder", "workflow", "explore"]


@dataclass(frozen=True)
class IterationBudget:
    """Resource tier for a range of autoloop iterations.

    allowed_modes: which plan modes may execute. None = no restriction.
    max_calls:    how many times this tier can reject a plan before the
                  budget degrades to "allow everything". None = never degrade.
    label:        short human-readable tag for logs / hints.
    """

    allowed_modes: tuple[str, ...] | None
    max_calls: int | None
    label: str

    def allows(self, mode: str | None) -> bool:
        if self.allowed_modes is None:
            return True
        return mode in self.allowed_modes


# 默认三档预算. 用 frozenset 之外的 tuple 是为了让 label 在 hint 里有序可读.
_OPEN = IterationBudget(allowed_modes=None, max_calls=None, label="open")
_MEDIUM = IterationBudget(
    allowed_modes=("coder", "explore"), max_calls=30, label="medium"
)
_LIGHT = IterationBudget(allowed_modes=("coder",), max_calls=20, label="light")


@dataclass(frozen=True)
class ProgressiveBudget:
    """Maps an iteration number to a resource tier.

    tiers: sorted (upper_bound, budget) pairs. for_iteration picks the first
    pair whose upper_bound >= n. Iterations past the last bound fall back to
    the open tier so a runaway loop doesn't hit an undefined budget.
    """

    tiers: tuple[tuple[int, IterationBudget], ...]

    def for_iteration(self, n: int) -> IterationBudget:
        for upper_bound, budget in self.tiers:
            if n <= upper_bound:
                return budget
        return _OPEN

    @classmethod
    def default(cls) -> ProgressiveBudget:
        return cls(
            tiers=(
                (10, _OPEN),
                (30, _MEDIUM),
                (50, _LIGHT),
            )
        )


__all__ = ["IterationBudget", "ProgressiveBudget", "PlanMode"]
