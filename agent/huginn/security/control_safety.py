"""安全执行闸门 — 把 死手/心跳 + 权威仲裁 编排到动作执行 (microduck architecture.md §6).

在"真实/远程/有损"执行器接入前的软件兑现:

- **权威仲裁** (``ControlAuthority``): 物理 > 本地 > 远程 > 自治. 权限不足的命令在
  execute 前被拒 (``CommandDeniedError``), 非 last-writer-wins; 高优先级可抢占.
- **死手** (``Deadman``): 命令停 / 超时即触发安全停机 (``SafetyStopError``),
  需显式 ``reset`` 恢复.
- **心跳**: 每次成功命令后 ``poke``, 表示"动作仍在进行 / 链路健康".

把真实执行器调用 (如 ``workspace.execute``) 包进 ``guard.command(source, fn)`` 即可,
与执行器本体解耦. 先于真实硬件落地, 保证"网络分区 / LLM 卡推理 / 设备休眠"时
不会无限等待一个已停止的命令源.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from huginn.security.control_authority import AuthoritySource, ControlAuthority, Deadman


class CommandDeniedError(Exception):
    """该来源无控制权 (被更高优先级抢占 / 未持权), 命令不执行."""


class SafetyStopError(Exception):
    """死手已触发 (命令长时间未到), 动作被安全停机拒绝. reset 后恢复."""


class ExecutionGuard:
    """把权威仲裁 + 死手/心跳编排成一层安全执行闸门."""

    def __init__(
        self,
        max_idle: float,
        *,
        on_safe_stop: Callable[[], None] | None = None,
        authority: ControlAuthority | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.authority = authority or ControlAuthority()
        self.stopped = False

        def _stall() -> None:
            self.stopped = True
            if on_safe_stop is not None:
                on_safe_stop()

        self.deadman = Deadman(max_idle, _stall, clock=clock)

    # ── 安全命令执行 ───────────────────────────────────────────
    def command(
        self,
        source: AuthoritySource,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """带权威仲裁 + 死手/心跳的动作执行.

        1) 已安全停机 → 拒 (SafetyStopError), 待 reset.
        2) 提交权威仲裁, 未获权 → 拒 (CommandDeniedError).
        3) 执行 ``fn``; 成功后喂心跳 (命令到达 = 链路健康).
        """
        if self.stopped:
            raise SafetyStopError("safety stop latched; reset to resume")
        if not self.authority.request(source):
            raise CommandDeniedError(
                f"command from '{source.name}' denied: owner='{self.authority.owner()}'"
            )
        try:
            return fn(*args, **kwargs)
        finally:
            self.deadman.poke()  # 命令成功执行 → 心跳 (链路健康)

    # ── 空闲巡检 / 恢复 ─────────────────────────────────────────
    def monitor(self) -> bool:
        """空闲巡检: 超时(无新命令)即触发安全停机. 返回当前是否已停机."""
        self.deadman.tick()
        return self.stopped

    def reset(self) -> None:
        """人工接管 / 链路恢复后重新武装."""
        self.stopped = False
        self.deadman.reset()
