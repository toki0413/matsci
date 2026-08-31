"""控制权威仲裁 + 死手/心跳 — microduck 安全层 (architecture.md §6) 的软件接口与仿真.

microduck §6 "safety and authority" 三条, 在 Huginn 接真实执行器前的最小确定实现:

- **Deadman / heartbeat**: 命令停或超时即自行安全停机 — 非协商选项 (网络会分区,
  LLM 会卡在推理中, 笔记本会休眠). 本模块 ``Deadman`` 是一次性触发 + 需显式 reset.
- **Intents, not raw actions**: 多来源都发"意图", 由确定层裁决, 执行层保留安全权威.
- **Explicit authority arbitration**: 物理控制器 > 本地 app > 远程 > 自治层, 定义好的
  优先级与抢占, 而非 last-writer-wins; 本地/物理可抢占远程主体. 本模块
  ``ControlAuthority`` = 单所有者 + 显式抢占.

纯 stdlib; 接真实/远程执行器时在 workspace 外层消费.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class AuthoritySource:
    """权威来源. ``rank`` 越小越优先 (microduck §6 的固定优先级)."""

    rank: int
    name: str


# microduck §6 的仲裁顺序: 物理/本地可抢占远程/自治.
PHYSICAL = AuthoritySource(0, "physical")
LOCAL = AuthoritySource(1, "local")
REMOTE = AuthoritySource(2, "remote")
AUTONOMOUS = AuthoritySource(3, "autonomous")


class ControlAuthority:
    """单所有者 + 显式优先级抢占仲裁 (非 last-writer-wins)."""

    _NO_OWNER = 10**9

    def __init__(self) -> None:
        self._owner: str | None = None
        self._owner_rank = self._NO_OWNER

    def request(self, src: AuthoritySource) -> bool:
        """请求权威. 更高优先级 (更小 rank) 或无主时获得; 同 rank 保持当前 holder.
        返回是否获得控制权."""
        if self._owner is None or src.rank < self._owner_rank:
            self._owner = src.name
            self._owner_rank = src.rank
            return True
        return False

    def release(self, src: AuthoritySource) -> None:
        if self._owner == src.name:
            self._owner = None
            self._owner_rank = self._NO_OWNER

    def owner(self) -> str | None:
        return self._owner


class Deadman:
    """死手: 超过 ``max_idle`` 未喂命令即触发一次安全停机 (simplify 一次性, 需 reset)."""

    def __init__(
        self,
        max_idle: float,
        on_stall: Callable[[], None],
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.max_idle = max_idle
        self.on_stall = on_stall
        self._clock = clock if clock is not None else _default_clock
        self._last: float | None = None
        self._tripped = False

    def poke(self) -> None:
        """喂命令 / 心跳 — 重置超时窗口 (命令仍在到达可认为动作在进行)."""
        self._last = self._clock()

    def tick(self) -> bool:
        """当前是否已停机 (超时触发一次 on_stall). 尚未收到过命令 (未武装) 不在
        判定范围内 — 停在刚上电、无人下命令的空闲态不算"命令中断"."""
        if self._tripped:
            return True
        if self._last is None:
            return False  # 尚未武装 (等第一条命令), 不启动超时.
        idle = self._clock() - self._last
        if idle > self.max_idle:
            self._tripped = True
            self.on_stall()
            return True
        return False

    def reset(self) -> None:
        """重新武装 (如远程命令链路已恢复 / 人工接管后)."""
        self._tripped = False
        self._last = None


def _default_clock() -> float:
    import time

    return time.monotonic()
