"""预算边缘 Checkpoint + 暂停 (BudgetPause) — 软停止而非硬杀.

长程任务到预算边缘时, 直接 kill 会让已投入的价值归零。BudgetPause 把"停止"改成
**软停止**: 保存 checkpoint → 记录一次 pause → 由上层 (策略/用户) 决定续投或放弃。
续投时带着前一个 checkpoint 接着跑, 不重头再来。

这是后端无关的轻量层: 实际持久化由调用方注入 `save_checkpoint` 可调用 (默认 No-op),
本模块只负责"暂停 = 保存状态 + 记一笔可恢复的账", 以及 resume 决策的可观测记录。

用法::

    from huginn.budget_pause import BudgetPauseHandler

    handler = BudgetPauseHandler(save_checkpoint=lambda cp: db.put(cp))
    p = handler.pause("s1", "超出阶段预算", {"iter": 12, "state": ...})
    # ... 用户/策略决定 ...
    handler.resume(p, decision="continue")   # 带着 checkpoint 续跑
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "BudgetPause",
    "BudgetPauseHandler",
    "get_budget_pause_handler",
    "reset_budget_pause_handler",
]


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


if __name__ == "__main__":
    # 自检
    saved: list[Any] = []
    h = BudgetPauseHandler(save_checkpoint=saved.append)
    p = h.pause("s1", "超出阶段预算", {"iter": 12})
    assert saved and saved[0] == {"iter": 12}, "checkpoint 应被保存"
    assert not p.resumed
    assert len(h.pending("s1")) == 1
    h.resume(p, decision="continue")
    assert p.resumed and p.resume_decision == "continue"
    assert len(h.pending("s1")) == 0
    assert h.list_pauses("s1")[0].checkpoint is not None
    print("budget_pause self-check passed")
