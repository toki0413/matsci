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
        return not self.deficits()

    def deficits(self) -> list[str]:
        """返回未达标项的可读描述, 给反完成审计拼消息用."""
        out: list[str] = []
        if self.iteration < self.required_floor.min_iterations:
            out.append(
                f"iteration={self.iteration} < min_iterations={self.required_floor.min_iterations}"
            )
        if self.families_explored < self.required_floor.min_method_families:
            out.append(
                f"families_explored={self.families_explored} "
                f"< min_method_families={self.required_floor.min_method_families}"
            )
        if self.live_components < self.required_floor.min_live_components:
            out.append(
                f"live_components={self.live_components} "
                f"< min_live_components={self.required_floor.min_live_components}"
            )
        return out


class PrematureConvergenceDetector:
    """过早收敛检测器 — 反完成审计的核心.

    agent 想返回结果时调 should_block_return, 未达标就强制继续.
    """

    def __init__(self, floor: MinEffortFloor | None = None) -> None:
        self._floor = floor or MinEffortFloor()
        self._dynamic = DynamicComponentFloor(self._floor)

    def check(
        self,
        iteration: int,
        families_explored: int,
        live_components: int,
        total_iterations: int = 10,
    ) -> EffortStatus:
        # 用动态下限替换 min_live_components, 其他两项保持原值
        dyn_components = self._dynamic.current_floor(iteration, total_iterations)
        effective_floor = replace(
            self._floor, min_live_components=dyn_components
        )
        return EffortStatus(
            iteration=iteration,
            families_explored=families_explored,
            live_components=live_components,
            required_floor=effective_floor,
        )

    def should_block_return(self, status: EffortStatus) -> tuple[bool, str]:
        """任一下限未达标 → 阻断返回. 返回 (是否阻断, 原因)."""
        gaps = status.deficits()
        if gaps:
            return True, "; ".join(gaps)
        return False, ""


# ── 自检 ─────────────────────────────────────────────────────────

def _selfcheck() -> None:
    # 1. MinEffortFloor 默认值
    f = MinEffortFloor()
    assert f.min_iterations == 3
    assert f.min_method_families == 3
    assert f.min_live_components == 2

    # 2. DynamicComponentFloor 三阶段衰减
    dyn = DynamicComponentFloor()
    assert dyn.current_floor(0, 9) == 4, "早期应强制发散到 4"
    assert dyn.current_floor(3, 9) == 2, "中期应允许部分收敛到 2"
    assert dyn.current_floor(6, 9) == 1, "后期应允许收敛到 1"
    # total=0 返回基础值
    assert dyn.current_floor(0, 0) == 2, "total=0 应返回基础值 2"

    # 3. EffortStatus.is_satisfied / deficits
    satisfied = EffortStatus(
        iteration=5,
        families_explored=4,
        live_components=3,
        required_floor=MinEffortFloor(),
    )
    assert satisfied.is_satisfied()
    assert satisfied.deficits() == []

    unsatisfied = EffortStatus(
        iteration=1,
        families_explored=2,
        live_components=1,
        required_floor=MinEffortFloor(),
    )
    assert not unsatisfied.is_satisfied()
    deficits = unsatisfied.deficits()
    assert len(deficits) == 3, f"三项都未达标应有 3 条 deficit, got {deficits}"

    # 4. PrematureConvergenceDetector.should_block_return
    det = PrematureConvergenceDetector()

    # 未达标 → 阻断
    early_status = det.check(
        iteration=0, families_explored=1, live_components=1, total_iterations=9
    )
    blocked, reason = det.should_block_return(early_status)
    assert blocked, "未达标应阻断返回"
    assert reason, "阻断时应给出原因"

    # 全部达标 → 不阻断
    # iteration=9 (后期, min_live_components=1), families=3, live_components=2
    ok_status = det.check(
        iteration=9, families_explored=3, live_components=2, total_iterations=9
    )
    blocked2, reason2 = det.should_block_return(ok_status)
    assert not blocked2, f"达标不应阻断, reason={reason2}"
    assert reason2 == ""

    print("depth_search selfcheck OK")


if __name__ == "__main__":
    _selfcheck()
