"""Monitor-hold 减负策略 — 对齐 Hermes "monitor 模式无变化则跳过 LLM".

长程科研 loop 里, 大量轮次其实是"没有新观测/新活动"的空转: 文件没变、结果没变、
环境静默。此时再驱动下一轮昂贵 LLM 推理 (hypothesize/plan) 是纯 token 浪费。

本模块把这条策略从"对话约定"固化成机器可判定的决策门:

  - ``MonitorHoldState``    : 追踪连续"空观测"次数 + 最近一次是否处理过.
  - ``decide_hold``         : 依 (本回合是否有新活动, 连续空观测次数, 阈值)
    返回是否应 hold (``skip``):
        * 有活动    → 不 hold (应继续推理)                    → ``False + reset``
        * 连续空观测 < 阈值 → 刚安静, 给一次处理机会           → ``False (不 hold)``
        * 连续空观测 ≥ 阈值 → 持久静默, 跳过昂贵推理           → ``True (hold)``
  - ``HOLD_THRESHOLD``      : 连续空观测达到几次才进入 hold.

设计原则: 纯函数 + 无副作用, 与认知循环解耦 — 调用方决定用不用; 绝不擅自改
动作合法性或回答内容, 只回答"这件事是否值得花一次 LLM 推理".
"""

from __future__ import annotations

from dataclasses import dataclass

# 连续空观测达到阈值才 hold. 阈值 2: 首轮空 ≠ 立刻 hold (避免误杀刚启动无信号),
# 但第二轮仍空 → 确实静默 → 跳过 LLM.
HOLD_THRESHOLD = 2


@dataclass
class MonitorHoldState:
    """连续空观测的计数 + 最近一次是否有活动."""

    empty_streak: int = 0
    last_had_activity: bool | None = None

    def to_dict(self) -> dict:
        return {"empty_streak": self.empty_streak, "last_had_activity": self.last_had_activity}


@dataclass(frozen=True)
class HoldDecision:
    """一次 hold 决策结果."""

    hold: bool          # True → 应跳过昂贵推理 (保持 silent)
    empty_streak: int   # 本回合的连续空观测次数
    reason: str


def decide_hold(
    state: MonitorHoldState,
    had_activity: bool,
    *,
    threshold: int = HOLD_THRESHOLD,
) -> HoldDecision:
    """判定本回合是否应 hold (跳过 LLM 推理).

    ``had_activity=False`` 表示本回合没有新观测/活动. 连续静止达 ``threshold``
    次即 hold. 一结束全靠状态更新 (无副作用); 调用方决定要不要真省那次调用.
    """
    if had_activity:
        state.empty_streak = 0
        state.last_had_activity = True
        return HoldDecision(
            hold=False,
            empty_streak=0,
            reason="new activity present, do not hold",
        )
    state.empty_streak += 1
    state.last_had_activity = False
    if state.empty_streak >= threshold:
        return HoldDecision(
            hold=True,
            empty_streak=state.empty_streak,
            reason=(
                f"no new observation for {state.empty_streak}x, "
                "skip expensive reasoning (monitor hold)"
            ),
        )
    return HoldDecision(
        hold=False,
        empty_streak=state.empty_streak,
        reason=(
            f"quiet for {state.empty_streak}x (below {threshold}), "
            "give one processing chance"
        ),
    )
