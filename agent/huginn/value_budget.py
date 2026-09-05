"""价值感知 + 阶段伸缩预算 (ValueBudget) — 平衡"性价比"与"长程任务不得不花".

核心矛盾: 精细控制想砍掉低价值花费, 长程任务又合法积累成本. 单一硬预算要么
提前误杀 (价值归零), 要么锁不住。ValueBudget 用两个正交能力来调和:

1. **阶段伸缩 (phase-discriminated)**: 同一本账, 不同阶段给不同预算系数。
   探索 (explore) 允许多花, 收敛 (converge) 收紧, 验证 (verify) 最紧。
   长程任务在探索期"该花的钱花得出去", 收敛期自动收紧。

2. **价值感知 (value-aware)**: 停止决策不只看花了多少, 看"边际价值"。
   已实现价值 / 已花成本 低于 `min_roi` 时 → 停止 (性价比不达标)。
   这比固定上限更符合"追求性价比"。

配合 CostLedger (统一账本) —— 此处通过参数注入 total_cost, 保持可测、解耦:

    from huginn.value_budget import ValueBudget, default_phase_ratios

    vb = ValueBudget(base_budget_usd=10.0, min_roi=1.0)
    decision, reason = vb.check("explore", total_cost_usd=3.0, delivered_value=5.0)
    # "allow" — 探索期, 预算几何, ROI 达标
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from huginn.env_defaults import get_float, registry_default

__all__ = [
    "default_phase_ratios",
    "ValueBudget",
    "PHASE_ORDER",
]

# 阶段伸缩系数: 数值 >1 放宽 (多给预算), <1 收紧. 缺省阶段用 1.0.
PHASE_ORDER = ("explore", "converge", "verify")


def default_phase_ratios() -> dict[str, float]:
    """默认阶段预算系数 — 探索放宽, 收敛收紧, 验证最紧."""
    return {
        "explore": 1.5,
        "converge": 1.0,
        "verify": 0.5,
    }


@dataclass
class ValueBudget:
    """价值感知 + 阶段伸缩预算."""

    base_budget_usd: float = field(
        default_factory=lambda: get_float(
            "HUGINN_VALUE_BUDGET_USD",
            default=float(registry_default("HUGINN_VALUE_BUDGET_USD") or 0.0),
        )
    )
    # 阶段 → 预算伸缩系数.
    phase_ratios: dict[str, float] = field(default_factory=default_phase_ratios)
    # 最低投入产出比 (value_per_usd): 已实现价值 / 已花成本 低于它 → 停止.
    # <=0 表示不启用价值感知 (纯预算控制).
    min_roi: float = field(
        default_factory=lambda: get_float(
            "HUGINN_VALUE_MIN_ROI",
            default=float(registry_default("HUGINN_VALUE_MIN_ROI") or 0.0),
        )
    )
    # 预算告警比例 (达到预算的该比例时 warn).
    warn_ratio: float = 0.8

    def effective_budget(self, phase: str) -> float:
        """某阶段的有效预算 = base * 该阶段系数 (缺省 1.0)."""
        if self.base_budget_usd <= 0:
            return float("inf")
        return self.base_budget_usd * self.phase_ratios.get(phase, 1.0)

    def value_ok(self, total_cost_usd: float, delivered_value: float) -> bool:
        """价值感知是否达标: ROI >= min_roi. 未启用 (<0) 恒 True."""
        if self.min_roi <= 0:
            return True
        if total_cost_usd <= 0:
            return True  # 还没花钱不判性价比
        return (delivered_value / total_cost_usd) >= self.min_roi

    def check(
        self,
        phase: str,
        total_cost_usd: float,
        delivered_value: float = 0.0,
    ) -> tuple[str, str]:
        """对当前阶段做三档判定 (allow / warn / deny).

        判定顺序:
          1. 成本 vs 该阶段有效预算: 超 → deny; 达 warn 线 → warn.
          2. 价值感知: 已花成本下 ROI 不达标 → deny (花得不值, 停止).
        返回 (decision, reason).
        """
        budget = self.effective_budget(phase)
        roi = (
            round(delivered_value / total_cost_usd, 3)
            if total_cost_usd > 0
            else float("inf")
        )
        base_reason = (
            f"阶段={phase} 已花=${total_cost_usd:.2f} 有效预算=${budget:.2f}"
            f" ROI={roi}"
        )

        if total_cost_usd >= budget:
            return "deny", f"{base_reason} — 超出阶段预算"
        if not self.value_ok(total_cost_usd, delivered_value):
            return "deny", f"{base_reason} — ROI 低于阈值 {self.min_roi}"
        if total_cost_usd >= budget * self.warn_ratio:
            return "warn", f"{base_reason} — 接近阶段预算"
        return "allow", f"{base_reason}"

    def allowed_phases(self, total_cost_usd: float) -> list[str]:
        """在给定累计成本下, 哪些阶段仍在其有效预算内 (可观测)."""
        return [
            p
            for p in PHASE_ORDER
            if total_cost_usd < self.effective_budget(p)
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_budget_usd": self.base_budget_usd,
            "phase_ratios": dict(self.phase_ratios),
            "min_roi": self.min_roi,
            "warn_ratio": self.warn_ratio,
        }


if __name__ == "__main__":
    # 自检
    vb = ValueBudget(base_budget_usd=10.0, min_roi=1.0)
    # 探索期: 预算 15, 花 3, ROI 5/3≈1.67 ≥1 → allow
    assert vb.check("explore", 3.0, 5.0)[0] == "allow"
    # 探索期花 12 (<15) 但 ROI 6/12=0.5 <1 → deny (价值不达标)
    assert vb.check("explore", 12.0, 6.0)[0] == "deny"
    # 验证期: 预算 5, 花 6 → deny (超预算)
    assert vb.check("verify", 6.0, 10.0)[0] == "deny"
    # 验证期: 预算 5, 花 4 (≥5*0.8=4) → warn
    assert vb.check("verify", 4.0, 4.0)[0] == "warn"
    # 未启用价值感知: ROI 归零阈值, 只按预算判
    vb2 = ValueBudget(base_budget_usd=10.0, min_roi=0.0)
    assert vb2.check("converge", 5.0, 0.0)[0] == "allow"
    assert vb2.check("converge", 11.0, 0.0)[0] == "deny"
    print("value_budget self-check passed")
