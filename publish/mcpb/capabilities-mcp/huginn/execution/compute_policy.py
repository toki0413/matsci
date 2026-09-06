"""M2 — 细粒度计算权限 + 预算 (按路由目标维度).

compute_router 决定"去哪算" (local/hpc/device/remote), 这里补"谁能在哪算、算多少":

- **按路由目标维度判权**: 高特权目标 (hpc / remote) 对非提升 (elevated) actor 默认
  要求审批 (requires_approval); 关时只审计不改行为.
- **compute 预算**: 每 actor 每时间窗内 heavy 调用 (DFT/MD/QC 等) 有配额; 超配额
  该次调用判 deny (不再偷偷跑).
- 所有决策在 orchestrator 热路径落 audit (compute_policy 事件).

通过统一的 FeatureFlags ``compute_policy`` 开关控制: 默认关 ⇒ 不改变现有执行;
显式开启后 enforce 才真正拦截. 与路由/后端选择分离, 职责单一.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# 高成本缩放族 → 视为 heavy (配额消费计入).
_HEAVY_SCALINGS = frozenset({"dft_md", "qc"})


@dataclass(frozen=True)
class PolicyVerdict:
    """一次策略判定的结果."""

    allowed: bool
    reason: str = ""
    requires_approval: bool = False


@dataclass
class _WindowCounter:
    """滑动时间窗计数器 (环形缓冲, 窗口内计数)."""

    window_seconds: float
    max_count: int
    _events: list[float] = field(default_factory=list)

    def try_consume(self) -> bool:
        """窗口内未超上限则记一次并返回 True, 否则 False."""
        now = time.monotonic()
        self._events = [t for t in self._events if now - t < self.window_seconds]
        if len(self._events) >= self.max_count:
            return False
        self._events.append(now)
        return True

    def count(self) -> int:
        now = time.monotonic()
        self._events = [t for t in self._events if now - t < self.window_seconds]
        return len(self._events)


class ComputePolicy:
    """按 (tool × target × actor × heavy) 判权 + per-actor 预算.

    默认: 非提升 actor 对 hpc/remote 目标要求审批; heavy 调用有窗口配额.
    满足"细粒度权限 + 审计": 决策粒度为一次工具调用, 决策理由可审计回放.
    """

    def __init__(
        self,
        *,
        elevated_actors: list[str] | None = None,
        heavy_window_seconds: float = 3600.0,
        max_heavy_per_window: int = 10,
        require_approval_targets: tuple[str, ...] = ("hpc", "remote"),
    ) -> None:
        self._elevated: set[str] = set(elevated_actors or ())
        self._require_approval_targets = set(require_approval_targets)
        self._budget: dict[str, _WindowCounter] = {}
        self._budget_cfg = (heavy_window_seconds, max_heavy_per_window)
        self._lock = threading.Lock()

    def is_heavy(self, scaling: str) -> bool:
        """某缩放族是否计入 heavy 配额."""
        return scaling in _HEAVY_SCALINGS

    def enforce(
        self,
        tool_name: str,
        target: str | None,
        actor: str,
        *,
        scaling: str,
    ) -> PolicyVerdict:
        """判定一次工具调用是否可执行; 可选标 requires_approval.

        Parameters
        ----------
        tool_name : 工具名
        target    : 路由目标 (local/hpc/device/remote 或 None=未路由)
        actor     : 调用方身份
        scaling   : 计算缩放族 (dft_md/qc/generic)
        """
        # 1) 按目标维度: 非提升 actor 访问高特权目标 → ask
        if target in self._require_approval_targets and actor not in self._elevated:
            return PolicyVerdict(
                allowed=True,
                requires_approval=True,
                reason=(
                    f"{target} target requires approval for {tool_name} "
                    f"(actor '{actor}' not elevated)"
                ),
            )

        # 2) heavy 配额
        if self.is_heavy(scaling) and actor not in self._elevated:
            w, m = self._budget_cfg
            counter = self._budget.setdefault(actor, _WindowCounter(w, m))
            with self._lock:
                if not counter.try_consume():
                    return PolicyVerdict(
                        allowed=False,
                        reason=(
                            f"heavy budget exhausted for actor '{actor}': "
                            f"{counter.count()}/{m} in window"
                        ),
                    )

        return PolicyVerdict(allowed=True, reason="allowed")


def _current_actor_from_context(context: dict) -> str:
    """从执行上下文提取 actor, 缺省 'system'."""
    actor = context.get("user") or context.get("actor")
    return str(actor) if actor else "system"
