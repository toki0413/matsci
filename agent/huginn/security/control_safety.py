"""安全执行闸门 — 把 死手/心跳 + 权威仲裁 + 策略清单 编排到动作执行.

在"真实/远程/有损"执行器接入前的软件兑现:

- **权威仲裁** (``ControlAuthority``): 物理 > 本地 > 远程 > 自治. 权限不足的命令在
  execute 前被拒 (``CommandDeniedError``), 非 last-writer-wins; 高优先级可抢占.
- **死手** (``Deadman``): 命令停 / 超时即触发安全停机 (``SafetyStopError``),
  需显式 ``reset`` 恢复.
- **心跳**: 每次成功命令后 ``poke``, 表示"动作仍在进行 / 链路健康".
- **策略清单** (``GatePolicy``): 把安全边界按 **绝对禁止 / 减少伤害 / 需授权** 三层
  维护成可枚举清单 (从 Fable 那套穷举护栏学来的粒径). 仲裁只决定"谁有权", 策略
  决定"这个动作值不值得/允不允许被任何人做" — 往往是更强的安全保证.

把真实执行器调用 (如 ``workspace.execute``) 包进 ``guard.command(source, fn, action=...)``
即可, 与执行器本体解耦. 先于真实硬件落地, 保证"网络分区 / LLM 卡推理 / 设备休眠"时
不会无限等待一个已停止的命令源.

三层策略 (``PolicyLevel``):

- ``ABSOLUTE_DENY`` : 该动作天然禁止, 任何人都不得执行 (如反向驱动使硬件过冲).
- ``REQUIRE_AUTHORITY`` : 需要足够强的控制源 (rank <= ``min_rank``) 才允许, 远程/自治
  被拒 — 高后果但偶需的动作.
- ``HARM_REDUCTION`` : 允许执行, 但记入 ``policy_log`` 供审计/降级标记.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from huginn.security.control_authority import (
    LOCAL,
    AuthoritySource,
    ControlAuthority,
    Deadman,
)


class CommandDeniedError(Exception):
    """该来源无控制权 (被更高优先级抢占 / 未持权), 命令不执行."""


class PolicyDeniedError(Exception):
    """策略清单拒绝: 动作被绝对禁止, 或来源不具备所需授权等级."""


class SafetyStopError(Exception):
    """死手已触发 (命令长时间未到), 动作被安全停机拒绝. reset 后恢复."""


class PolicyLevel(Enum):
    """策略分层 — 从 Fable 穷举护栏学的粒径分级."""

    ABSOLUTE_DENY = "absolute_deny"
    HARM_REDUCTION = "harm_reduction"
    REQUIRE_AUTHORITY = "require_authority"


@dataclass(frozen=True)
class SafetyRule:
    """一条策略清单项.

    - ``action``    : 命中的动作名; ``None`` = 通配 (对任何动作生效).
    - ``level``     : 分层策略.
    - ``min_rank``  : 仅 ``REQUIRE_AUTHORITY`` 用; 来源 rank 须 <= 该值 (rank 小=更强).
    - ``note``      : 人类可读理由 (同样作为判定信息暴露给上层).
    """

    level: PolicyLevel
    note: str = ""
    action: str | None = None
    min_rank: int = LOCAL.rank


@dataclass(frozen=True)
class PolicyVerdict:
    """一次策略判定结果."""

    allowed: bool
    level: PolicyLevel | None = None
    note: str = ""


class GatePolicy:
    """可枚举的分层安全策略清单. 先验"动作是否允许", 独立于权威仲裁."""

    def __init__(self, rules: list[SafetyRule] | None = None) -> None:
        self.rules: list[SafetyRule] = list(rules or [])

    def add(self, rule: SafetyRule) -> None:
        self.rules.append(rule)

    def check(self, source: AuthoritySource, action: str | None) -> PolicyVerdict:
        """判定 ``action`` 对 ``source`` 是否被策略许可.

        优先级: 任一绝对禁止命中 → 拒; 需授权项若来源 rank 不足 → 拒;
        否则若有减少伤害命中 → 放行并降级标记; 全通过 → 放行.
        """
        denied_note: str | None = None
        authority_note: str | None = None
        harm_notes: list[str] = []
        for r in self.rules:
            if r.action is not None and r.action != action:
                continue
            if r.level is PolicyLevel.ABSOLUTE_DENY:
                denied_note = denied_note or (r.note or f"absolute-deny:{action}")
            elif r.level is PolicyLevel.REQUIRE_AUTHORITY:
                if source.rank > r.min_rank:
                    authority_note = authority_note or (
                        r.note or f"requires-authority:{action}"
                    )
            elif r.level is PolicyLevel.HARM_REDUCTION:
                harm_notes.append(r.note or "harm-reduction")
        if denied_note is not None:
            return PolicyVerdict(False, PolicyLevel.ABSOLUTE_DENY, denied_note)
        if authority_note is not None:
            return PolicyVerdict(False, PolicyLevel.REQUIRE_AUTHORITY, authority_note)
        if harm_notes:
            return PolicyVerdict(
                True, PolicyLevel.HARM_REDUCTION, "; ".join(harm_notes)
            )
        return PolicyVerdict(True)


class ExecutionGuard:
    """把权威仲裁 + 死手/心跳 + 策略清单编排成一层安全执行闸门."""

    def __init__(
        self,
        max_idle: float,
        *,
        on_safe_stop: Callable[[], None] | None = None,
        authority: ControlAuthority | None = None,
        clock: Callable[[], float] | None = None,
        policy: GatePolicy | None = None,
    ) -> None:
        self.authority = authority or ControlAuthority()
        self.stopped = False
        self.policy = policy
        # HARM_REDUCTION 命中的动作记这里, 供审计/降级标记 (从 Fable 穷举护栏学的粒度).
        self.policy_log: list[dict[str, Any]] = []

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
        action: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """带策略清单 + 权威仲裁 + 死手/心跳的动作执行.

        1) 已安全停机 → 拒 (SafetyStopError), 待 reset.
        2) ``policy`` 不足时按分层策略判定: 绝对禁止 / 授权不足 → ``PolicyDeniedError``;
           减少伤害命中 → 放行但记入 ``policy_log``. (不传 ``action`` 且无通配规则时跳过.)
        3) 提交权威仲裁, 未获权 → 拒 (CommandDeniedError).
        4) 执行 ``fn``; 成功后喂心跳 (命令到达 = 链路健康).
        """
        if self.stopped:
            raise SafetyStopError("safety stop latched; reset to resume")
        if self.policy is not None:
            verdict = self.policy.check(source, action)
            if not verdict.allowed:
                raise PolicyDeniedError(
                    verdict.note or f"policy denied action {action!r}"
                )
            if verdict.level is PolicyLevel.HARM_REDUCTION:
                self.policy_log.append(
                    {"action": action, "note": verdict.note, "source": source.name}
                )
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
