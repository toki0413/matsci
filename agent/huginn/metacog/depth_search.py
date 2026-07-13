"""深度搜索 — 最小努力下限, 对抗 LLM 的快速收敛偏差.

LLM 天然倾向尽快给出"看起来完整"的答案以结束交互. 这个模块给反完成
审计提供一个硬性下限: 在迭代轮数 / 方法族覆盖 / 存活连通分量三项上任一
不达标, 就阻断 agent 返回, 强制继续探索.

和 failure_modes 里 first-principles-violation 的 warn-then-force-proceed
不同, 这三项是硬性的——防的是认知偏差不是物理风险, 不允许 force proceed.

三个下限的含义:
- min_iterations:      对应 prompt "至少 8 小时" 的轮数下限
- min_method_families: 至少探索过几个思想本质不同的方法族
- min_live_components: 至少还有几个存活假设 (HypothesisGraph 连通分量)

DynamicComponentFloor 给 min_live_components 加阶段衰减: 早期强制发散,
后期允许收敛到单一最优. iteration / families 来自外部计数, live_components
来自 HypothesisGraph.connected_components(), 本模块不依赖它们的具体实现.

本模块暂未接入 engine, 下一阶段才接入.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class MinEffortFloor:
    """最小努力下限 (硬性, 不可 force proceed)."""

    min_iterations: int = 3
    min_method_families: int = 3
    min_live_components: int = 2


class DynamicComponentFloor:
    """min_live_components 的三阶段衰减策略.

    早期强制发散 (4), 中期允许部分收敛 (2), 后期允许收敛到单一最优 (1).
    total=0 时返回基础值, 不做阶段判断.
    """

    EARLY_FLOOR = 4
    MID_FLOOR = 2
    LATE_FLOOR = 1

    def __init__(self, base_floor: MinEffortFloor | None = None) -> None:
        self._base = base_floor or MinEffortFloor()

    def current_floor(self, iteration: int, total: int) -> int:
        if total <= 0:
            return self._base.min_live_components
        third = total / 3
        if iteration < third:
            return self.EARLY_FLOOR
        if iteration < 2 * third:
            return self.MID_FLOOR
        return self.LATE_FLOOR


@dataclass
class EffortStatus:
    """当前努力状态评估快照."""

    iteration: int
    families_explored: int
    live_components: int
    required_floor: MinEffortFloor

    def is_satisfied(self) -> bool:
        return not