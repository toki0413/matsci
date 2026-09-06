"""预算生命周期管理: 软停止(SoftPause) + 审批续投(Approval) 一体.

长程任务到预算边缘时, 直接 kill 会让已投入的价值归零。本模块把"停止"做成
**软停止**: 保存 checkpoint → 记一笔暂停 → 由上层(策略/用户)决定续投或放弃;
续投时带着 checkpoint 接着跑。同时承接预算审批续投: off(默认) / auto(有限续) /
gui(人工批)。

对接 autoloop/budget.TokenBudget 的 soft_limit(硬上限 80%) 与
engine_control._track_llm_usage。off 默认 = 保持现有硬刹车行为不打扰。

save_checkpoint 由调用方注入(默认 No-op), 本模块只负责
"暂停 = 保存状态 + 记一笔可恢复的账" + resume 决策可观测记录。

设计 (ponytail):
- off (默认): 完全不动, 维持现有"硬刹车即停"行为, 软限制仍只作信号。
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
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "BudgetPause",
    "BudgetPauseHandler",
    "get_budget_pause_handler",
    "reset_budget_pause_handler",
    # 审批续投
    "BudgetApprovalController",
    "get_approval_mode",
    "get_approval_controller",
    "register_budget_approval_callback",
]


# ── 软停止: 预算边缘 checkpoint + 暂停 ────────────────────────────────


@dataclass
class BudgetPause:
    """一次预算边缘的软停止记录."""

    session_id: str
    reason: str
    checkpoint: Any  # 交给 save_checkpoint 持久化的 payload
    created_at: float = field(default_factory=time.time)
    resumed: bool = False
    resume_decision: str = ""  # "continue" | "abort"
    resumed_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "reason": self.reason,
            "created_at": self.created_at,
            "resumed": self.resumed,
            "resume_decision": self.resume_decision,
            "resumed_at": self.resumed_at,
            # checkpoint 不序列化进观测 (可能很大/不可 JSON), 只给标记.
            "has_checkpoint": self.checkpoint is not None,
        }


class BudgetPauseHandler:
    """进程内暂停记录器. 保存动作注入, 状态可观测."""

    def __init__(
        self,
        save_checkpoint: Callable[[Any], Any] | None = None,
    ) -> None:
        self._save_checkpoint = save_checkpoint or (lambda cp: None)
        self._pauses: list[BudgetPause] = []

    def pause(
        self,
        session_id: str,
        reason: str,
        checkpoint_payload: Any = None,
    ) -> BudgetPause:
        """预算边缘软停止: 保存 checkpoint + 记录一次 pause."""
        self._save_checkpoint(checkpoint_payload)
        p = BudgetPause(
            session_id=session_id,
            reason=reason,
            checkpoint=checkpoint_payload,
        )
        self._pauses.append(p)
        return p

    def resume(
        self,
        pause: BudgetPause,
        *,
        decision: str = "continue",
    ) -> BudgetPause:
        """决定续投 (continue) 或放弃 (abort). 返回更新后的 pause."""
        pause.resumed = True
        pause.resume_decision = decision
        pause.resumed_at = time.time()
        return pause

    def list_pauses(self, session_id: str | None = None) -> list[BudgetPause]:
        """列出暂停记录 (可过滤 session)."""
        if session_id is None:
            return list(self._pauses)
        return [p for p in self._pauses if p.session_id == session_id]

    def pending(self, session_id: str | None = None) -> list[BudgetPause]:
        """未决 (未 resume) 的暂停."""
        return [p for p in self.list_pauses(session_id) if not p.resumed]

    def clear(self, session_id: str | None = None) -> None:
        """清空 (测试/会话收尾)."""
        if session_id is None:
            self._pauses.clear()
        else:
            self._pauses = [p for p in self._pauses if p.session_id != session_id]


# 进程级单例.
_handler: BudgetPauseHandler | None = None


def get_budget_pause_handler() -> BudgetPauseHandler:
    """全局单例."""
    global _handler
    if _handler is None:
        _handler = BudgetPauseHandler()
    return _handler


def reset_budget_pause_handler() -> None:
    """测试辅助: 重建单例."""
    global _handler
    _handler = None


# ── 审批续投: 软限制快用完时请求续投 ────────────────────────────────

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


def get_approval_controller(budget: Any) -> BudgetApprovalController:
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

    async def on_tokens_used_async(
        self,
        human_decide: Callable[[str, str], Awaitable[bool]] | None = None,
    ) -> str:
        """GUI 模式的异步审批路径: 越过软限制时用注入的 human_decide 问用户.

        与同步 on_tokens_used 的区别: gui 模式不再"无 callback 降级为 auto",
        而是真挂起等用户 (Inbox 通道)。human_decide(question, reason) -> bool:
        批准续投返回 True, 否则 False。engine 侧把它接到
        _await_human_decision_via_inbox 上, 由用户在任意 surface 决定。

        decision 语义与同步版一致: continue / renewed / abort。
        """
        if self._mode == "off":
            return "continue"
        b = self._budget
        if not b.is_over_soft():
            return "continue"
        if b.renewals_left() <= 0:
            # 额度用尽: gui 明确停, auto/无等待路径交硬刹车兜住.
            if self._mode == "gui":
                return "abort"
            if b.current_tokens > b.hard_limit_tokens:
                return "abort"
            return "continue"

        def _decide(question: str, reason: str) -> bool:
            approved = self._callback(question, reason) if self._callback else True
            return approved

        if human_decide is not None:
            # Inbox 人工审批优先: 真挂起等用户, 不问同步 callback.
            approved = await human_decide(
                "预算快用完, 是否批准续投额度?",
                f"当前 tokens {b.current_tokens}, 超软限制 {b.soft_limit_tokens}",
            )
            if approved:
                b.renew()
                return "renewed"
            return "abort"

        if self._mode == "auto":
            b.renew()
            return "renewed"
        # gui + 无 human_decide + 有同步 callback: 回退同步语义.
        if _decide("budget_renew", "预算快用完, 是否批准续投额度?"):
            b.renew()
            return "renewed"
        return "abort"


if __name__ == "__main__":
    from huginn.autoloop.budget import BudgetExhausted, TokenBudget

    saved: list[Any] = []

    # 软停止自检
    h = BudgetPauseHandler(save_checkpoint=saved.append)
    p = h.pause("s1", "超出阶段预算", {"iter": 12})
    assert saved and saved[0] == {"iter": 12}, "checkpoint 应被保存"
    assert not p.resumed
    assert len(h.pending("s1")) == 1
    h.resume(p, decision="continue")
    assert p.resumed and p.resume_decision == "continue"
    assert len(h.pending("s1")) == 0
    print("soft-pause self-check passed ✓")

    # 审批续投自检: auto 有限续到上限, 之后被硬刹车兜住.
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
        assert b.soft_limit_tokens == int(b.hard_limit_tokens * 0.8)
        print(f"{mode}: {renewals} renewals before hard brake ✓")

    # gui 拒绝: abort 且不耗续投额度
    b = TokenBudget()
    b.hard_limit_tokens = 1000
    b._recompute_soft()
    b.current_tokens = 900  # 已超 soft=800, 未超 hard=1000
    ctl = BudgetApprovalController(b, mode="gui", callback=lambda _n, _r: False)
    assert ctl.on_tokens_used() == "abort"
    assert b.renewals_left() == b.max_renewals
    print("gui deny -> abort, renewals intact ✓")

    # off: 不激活
    b = TokenBudget()
    b.hard_limit_tokens = 1000
    b._recompute_soft()
    ctl = BudgetApprovalController(b, mode="off")
    assert not ctl.is_active()
    print("off mode inactive ✓")
    print("budget_pause self-check passed")
