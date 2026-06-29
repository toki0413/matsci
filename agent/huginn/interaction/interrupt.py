"""中途干预管理 —— 用户可以在 agent 执行过程中暂停 / 恢复 / 取消 / 修改.

设计目标:
- 无干预时零开销: check_interrupt 只查一个 dict, 没事件直接返回 None.
- 不破坏现有 chat 流程: agent loop 在每轮工具调用前主动调一次 check_interrupt,
  没事件就当什么都没发生, 继续往下走.
- 跨请求传递: 干预事件由 HTTP 路由提交, 存在进程内 dict 里, agent loop
  在另一个协程里读. 同一 thread_id 关联同一个 agent 会话.

四种干预类型:
- pause:  暂停. agent loop 命中后阻塞在 check_interrupt 上, 直到 resume.
- resume: 恢复. 清掉 pause 状态, 让 agent 继续.
- cancel: 取消. agent loop 抛 InterruptCancelled, 终止本轮 chat.
- modify: 修改. 把用户给的 message 注入下一轮 LLM 输入, agent 继续.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class InterruptEvent:
    """一条干预事件. type 决定 agent loop 怎么响应."""

    type: str  # pause | resume | cancel | modify
    message: str = ""
    timestamp: float = field(default_factory=time.time)
    # 调用方塞的额外元数据, 比如前端给的用户身份 / 来源页面
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.type = self.type.lower()
        if self.type not in ("pause", "resume", "cancel", "modify"):
            raise ValueError(
                f"unknown interrupt type: {self.type} "
                f"(expected pause/resume/cancel/modify)"
            )


class InterruptCancelled(Exception):
    """用户点了取消. agent loop 捕获后跳出 chat 循环."""


class InterruptManager:
    """管理每个 thread 的干预事件队列.

    内部按 thread_id 维护两份状态:
    - _pending: 待处理的事件队列 (FIFO). check_interrupt 从这里取.
    - _paused: 是否处于暂停态. pause 后置位, resume 后清掉.

    暂停态用 asyncio.Event 实现, 同一事件循环里的 agent loop 直接 await,
    跨请求的 resume 路由调 event.set() 唤醒. 没有运行中的 loop 时退化为
    轮询检查, 但正常 FastAPI 路径不会走到这里.
    """

    def __init__(self) -> None:
        self._pending: dict[str, list[InterruptEvent]] = {}
        self._paused: dict[str, asyncio.Event] = {}
        # threading.Lock 而不是 asyncio.Lock: submit_interrupt 从 HTTP
        # 路由进来, 可能在不同线程; check_interrupt 在 agent loop 线程.
        # 锁持有时间极短 (只动 dict), 不会阻塞事件循环.
        self._lock = threading.Lock()

    # ── 用户侧 (HTTP 路由调用) ─────────────────────────────────

    def submit_interrupt(self, thread_id: str, event: InterruptEvent) -> None:
        """用户提交一条干预. 立即生效 (pause 置位, resume 唤醒, 等)."""
        with self._lock:
            if event.type == "resume":
                # resume 不进队列, 直接清 pause 态
                self._paused.pop(thread_id, None)
                return
            if event.type == "pause":
                # pause 也不进队列, 直接置位让 agent loop 阻塞
                self._paused[thread_id] = asyncio.Event()
                return
            # cancel / modify 进队列, agent loop 下一轮 check 时消费
            self._pending.setdefault(thread_id, []).append(event)

    def clear_interrupt(self, thread_id: str) -> None:
        """清掉一个 thread 的所有干预状态. 通常在 chat 结束时调."""
        with self._lock:
            self._pending.pop(thread_id, None)
            self._paused.pop(thread_id, None)

    # ── agent loop 侧 ──────────────────────────────────────────

    def check_interrupt(self, thread_id: str) -> Optional[InterruptEvent]:
        """非阻塞地查一次干预. 没事件返回 None.

        注意: pause 不走队列, 这里只返回 cancel / modify. pause 的阻塞
        在 wait_if_paused 里单独处理, 让调用方显式选择要不要等.
        """
        with self._lock:
            queue = self._pending.get(thread_id)
            if not queue:
                return None
            return queue.pop(0)

    async def wait_if_paused(self, thread_id: str) -> None:
        """如果当前 thread 处于 pause 态, 阻塞到 resume 进来.

        没有运行中的事件循环时直接返回 (退化为非阻塞, 由调用方自己轮询).
        正常 FastAPI / agent loop 路径都有 loop, 这里能正确挂起.
        """
        evt: Optional[asyncio.Event]
        with self._lock:
            evt = self._paused.get(thread_id)
        if evt is None:
            return
        # 拿到的是 asyncio.Event, 跨协程共享. resume 路由调 set() 唤醒.
        # 这里要小心: evt 可能在我们 await 期间被 clear/resume 掉,
        # 但 asyncio.Event.wait() 本身是幂等的, 已 set 的会立刻返回.
        try:
            await asyncio.wait_for(evt.wait(), timeout=None)
        except asyncio.CancelledError:
            # agent loop 被取消时也跟着退, 别吞掉 CancelledError
            raise

    def is_paused(self, thread_id: str) -> bool:
        """查是否处于 pause 态. 给前端 status 路由用."""
        with self._lock:
            return thread_id in self._paused

    def status(self, thread_id: str) -> dict[str, Any]:
        """返回当前 thread 的干预状态摘要. 给 /interrupt/status 用."""
        with self._lock:
            pending = list(self._pending.get(thread_id, []))
            paused = thread_id in self._paused
        return {
            "thread_id": thread_id,
            "paused": paused,
            "pending_count": len(pending),
            "pending_types": [e.type for e in pending],
            "last_message": pending[-1].message if pending else "",
        }


# ── 进程级单例 ──────────────────────────────────────────────
#
# 路由层和 agent loop 都要拿到同一个 InterruptManager. 用模块级单例
# 最简单, 不引入额外依赖. 多 worker 部署时每个 worker 各自一份,
# 干预只能影响当前 worker 上的 agent loop —— 这够用了, 因为同一
# thread_id 的 chat 请求通常落在同一个 worker.

_singleton: InterruptManager | None = None
_singleton_lock = threading.Lock()


def get_interrupt_manager() -> InterruptManager:
    """拿进程级 InterruptManager 单例."""
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = InterruptManager()
    return _singleton
