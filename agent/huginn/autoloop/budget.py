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

import os
from dataclasses import dataclass, field
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


class BudgetExhausted(Exception):  # noqa: N818
    """Token/cost 预算耗尽 — agent loop 应捕获并优雅停止.

    硬上限 (hard_limit_tokens / hard_limit_cost) 触发; 软上限只算信号不算刹车.
    """


@dataclass
class TokenBudget:
    """LLM 调用的 token + cost 硬刹车预算.

    每次 LLM 调用后调 update(usage_tokens, usage_cost) 累加. 超硬上限抛
    BudgetExhausted, agent loop 捕获后 save_engine_state 保存进度再停.
    默认值 (10M tokens / $50) 足够大多数短任务, 长任务/极限模式调 env 覆盖.

    ponytail: soft_limit 只存不用 (留给上层 warning), 硬上限才刹车.
    """

    hard_limit_tokens: int = field(
        default_factory=lambda: int(os.environ.get("HUGINN_TOKEN_BUDGET", "10000000"))
    )
    soft_limit_tokens: int = -1  # <0 = 取 hard_limit 的 80%
    hard_limit_cost: float = field(
        default_factory=lambda: float(os.environ.get("HUGINN_COST_BUDGET", "50.0"))
    )
    current_tokens: int = 0
    current_cost: float = 0.0
    # 软限制续投 (budget_pause 接线). 硬刹车是 update 抛 BudgetExhausted;
    # 软限制只在这里记状态, 由上层决定 auto 有限续 / GUI 人工批 / 忽略(off).
    max_renewals: int = field(
        default_factory=lambda: int(os.environ.get("HUGINN_BUDGET_MAX_RENEWALS", "3"))
    )
    _renewals: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        if self.soft_limit_tokens < 0:
            self.soft_limit_tokens = int(self.hard_limit_tokens * 0.8)

    def update(self, usage_tokens: int, usage_cost: float) -> None:
        """累加用量; 超任一硬上限抛 BudgetExhausted."""
        self.current_tokens += int(usage_tokens)
        self.current_cost += float(usage_cost)
        if self.is_exhausted():
            raise BudgetExhausted(
                f"token budget exhausted: "
                f"tokens={self.current_tokens}/{self.hard_limit_tokens} "
                f"cost={self.current_cost:.4f}/{self.hard_limit_cost:.4f}"
            )

    def is_exhausted(self) -> bool:
        """超任一硬上限返回 True."""
        return (
            self.current_tokens > self.hard_limit_tokens
            or self.current_cost > self.hard_limit_cost
        )

    # ── 软限制 (快用完预警) ──────────────────────────────────────────
    def is_over_soft(self) -> bool:
        """是否已超过软限制(硬上限 80%). soft_limit_tokens 是 token 维."""
        return self.current_tokens > self.soft_limit_tokens

    def renewals_left(self) -> int:
        """还剩几次续投额度 (auto + GUI 人工批共用同一个额度)."""
        return self.max_renewals - self._renewals

    def renew(self) -> bool:
        """执行一次续投: 硬上限×1.5 并按比例重算软限制. 额度用尽返回 False.

        auto 自动续与 GUI 人工批准都走这里, 共用 max_renewals, 防止无头任务
        无限烧钱。续满后不再续, 下次 update 自然被硬刹车兜住。
        ponytail: 固定 1.5 倍增, 不为每种模式做可配置放大系数.
        """
        if self._renewals >= self.max_renewals:
            return False
        self._renewals += 1
        self.hard_limit_tokens = int(self.hard_limit_tokens * 1.5)
        self._recompute_soft()
        return True

    def _recompute_soft(self) -> None:
        """按新的 hard_limit 重算软限制(默认 80%)."""
        self.soft_limit_tokens = int(self.hard_limit_tokens * 0.8)


__all__ = ["IterationBudget", "ProgressiveBudget", "PlanMode", "TokenBudget", "BudgetExhausted"]
