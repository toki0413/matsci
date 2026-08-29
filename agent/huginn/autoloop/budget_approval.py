"""预算软限制审批 — 快用完时请求续投(auto 有限续 / GUI 人工批 / off).

对接 autoloop/budget.TokenBudget 的 soft_limit(硬上限 80%) 与
engine_control._track_llm_usage 的每次调用累加点。

设计 (ponytail):
- off  (默认): 完全不动, 维持现有"硬刹车即停"行为, 软限制仍只作信号。
- auto: 每次超过软限制自动续投一次(限额×1.5), 最多 max_renewals 次;
        额度用尽后任其被硬刹车兜住, 不无限烧钱。
- gui: 走注入的 approval_callback(name, reason) -> bool, 批准才续投;
        拒绝返回 abort -> 上层保存状态后优雅停。无 callback 时降级为 auto,
        避免无头任务堵死。

GUI 的人机交互不在此模块阻塞: callback 是同步"状态查询", 由桌面端在两次
调用之间异步完成弹窗与批复, 回调只读用户当前决定。
"""

from __future__ import annotations

import os
from typing import Any, Callable

__all__ = [
    "BudgetApprovalController",
    "get_approval_controller",
    "register_budget_approval_callback",
]

# 桌面端 / web 端注册的人工审批回调 (进程级单例).
_callback: Callable[[str, str], bool] | None = None


def register_budget_approval_callback(
    fn: Callable[[str, str], bool] | None,
) -> None:
    """注册/注销人工审批回调. fn(name, reason) -> bool(批准续投).

    无头 CLI / bench 不注册, 走 auto; 桌面端把事件推给用户后再注册。
    """
    global _callback
    _callback = fn


def get_approval_mode() -> str:
    """返回 HUGINN_BUDGET_APPROVAL 解析结果: off | auto | gui."""
    mode = os.environ.get("HUGINN_BUDGET_APPROVAL", "off").strip().lower()
    return mode if mode in ("off", "auto", "gui") else "off"


def get_approval_controller(budget: Any) -> "BudgetApprovalController":
    """构造当前模式的审批控制器(读 env + 全局 callback)."""
    return BudgetApprovalController(budget, mode=get_approval_mode(), callback=_callback)


class BudgetApprovalController:
    """在每次 LLM 调用后裁决预算软限制是续投还是停.

    decision 语义:
      "continue"  常规推进(off / 未超软限制 / auto 到顶但未达硬刹车)
      "renewed"   本次执行了一次续投(限额已被刷新)
      "abort"     gui 用户拒绝 / 续投额度用尽(应保存状态并优雅停止)
    """

    def __init__(
        self,
        budget: Any,
        mode: str = "off",
        callback: Callable[[str, str], bool] | None = None,
    ) -> None:
        self._budget = budget
        self._mode = mode if mode in ("auto", "gui") else "off"
        self._callback = callback

    def is_active(self) -> bool:
        """是否启用了续投逻辑(off 时 False, 完全交硬刹车)."""
        return self._mode != "off"

    def on_tokens_used(self) -> str:
        """每次 LLM 调用后调用; 返回上方的 decision."""
        if self._mode == "off":
            return "continue"
        b = self._budget
        if not b.is_over_soft():
            return "continue"
        if b.renewals_left() <= 0:
            # 续投额度用尽: gui 明确告知到顶; auto 直接交硬刹车兜住.
            return "abort" if self._mode == "gui" else "continue"
        if self._mode == "auto":
            b.renew()
            return "renewed"
        # gui: 无 callback 时降级为 auto(不堵死), 有则问用户.
        approved = self._callback("budget_renew", "预算快用完, 是否批准续投额度?") if self._callback else True
        if approved:
            b.renew()
            return "renewed"
        return "abort"


if __name__ == "__main__":
    from huginn.autoloop.budget import BudgetExhausted, TokenBudget

    # auto 有限续自检: 超软限制续到上限, 之后被硬刹车兜住.
    for mode in ("auto", "gui"):
        b = TokenBudget()
        b.hard_limit_tokens = 1000
        b._recompute_soft()  # soft = 800
        ctl = BudgetApprovalController(b, mode=mode, callback=lambda _n, _r: True)
        renewals = 0
        while True:
            try:
                d = ctl.on_tokens_used()
                if d == "renewed":
                    renewals += 1
                b.update(500, 0.0)  # 每次 +500, 会反复跨软限制并最终超硬刹车
            except BudgetExhausted:
                break
        assert renewals == b.max_renewals, f"{mode}: renewals={renewals}"
        assert b.renewals_left() == 0
        # 续投后硬上限应单调递增, 软限制按 80% 重算低于当前硬上限
        assert b.soft_limit_tokens == int(b.hard_limit_tokens * 0.8)
        print(f"{mode}: {renewals} renewals before hard brake ✓")

    # gui 拒绝自检
    b = TokenBudget()
    b.hard_limit_tokens = 1000
    b._recompute_soft()
    b.current_tokens = 900  # 已超 soft=800, 未超 hard=1000
    ctl = BudgetApprovalController(b, mode="gui", callback=lambda _n, _r: False)
    assert ctl.on_tokens_used() == "abort"
    print("gui deny -> abort ✓")

    # gui 拒绝后再超软限制, 该次调用不会续(renewal 计数未动)
    assert b.renewals_left() == b.max_renewals
    print("gui deny leaves renewals intact ✓")

    # off 自检: 不续投, 交硬刹车
    b = TokenBudget()
    b.hard_limit_tokens = 1000
    b._recompute_soft()
    ctl = BudgetApprovalController(b, mode="off")
    assert not ctl.is_active()
    print("off mode inactive ✓")
    print("budget_approval self-check passed")