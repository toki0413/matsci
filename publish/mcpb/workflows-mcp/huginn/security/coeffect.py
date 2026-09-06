"""Reactive co-effects — 空间可组合性 (Cordis 论文).

对应论文的**空间可组合性**: 组件之间依赖可声明、可管理. 每个组件声明
``provides`` (我产生什么) 与 ``requires`` (我依赖什么); 运行时上下文里某个
produce 状态变化时, 按依赖规范通知相关组件 **激活 / 停用 / 保持中立** —
激活/停用决定由依赖推导, 正确性从"开发者手动接线"变成"声明即保证".

时间可组合性 (RevertibleContext) 回答"副作用如何撤销", 本模块回答
"依赖如何声明与解绑, 组件何时激活动/停用". 两者正交互补.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# 通知动作: 依赖满足 → 激活; 依赖消失 → 停用; 与本次变化无关 → 保持中立.
ACTIVATE = "activate"
DEACTIVATE = "deactivate"
NEUTRAL = "neutral"

# on_change 的回调签名: (key, action) -> None.
OnChange = Callable[[str, str], None]


@dataclass
class CoComponent:
    """一个可组合组件: 声明它产生和依赖的上下文键."""

    id: str
    provides: set[str] = field(default_factory=set)
    requires: set[str] = field(default_factory=set)
    on_change: OnChange | None = None
    # 当前是否激活: 所有 requires 都可用 (依赖满足) 才为真.
    active: bool = field(default=False, init=False)

    def evaluate(self, is_available: Callable[[str], bool]) -> bool:
        """根据依赖可用性推导该组件是否应激活."""
        if not self.requires:
            return True
        return all(is_available(k) for k in self.requires)


class CoEffectRegistry:
    """声明式依赖注册表 + 反应式通知中枢 (级联传播).

    空间可组合性: 组件通过 ``declare`` 声明依赖, 运行时 ``set_available`` /
    ``unbind`` 驱动上下文变化, 依赖者被自动激活/停用, 无关组件保持中立.
    组件的停用会级联使其 produces 一同下线 (依赖链传递).
    """

    def __init__(self) -> None:
        self._components: dict[str, CoComponent] = {}
        # key -> 当前是否可用 (是否仍有活跃 provider)
        self._available: dict[str, bool] = {}
        # key -> 提供它的组件 id 集合
        self._providers: dict[str, set[str]] = defaultdict(set)
        # key -> 依赖它的组件 id 集合 (requires 反查)
        self._dependents: dict[str, set[str]] = defaultdict(set)

    # ── 注册/解绑 ────────────────────────────────────────────────
    def declare(
        self,
        component_id: str,
        *,
        provides: set[str] | tuple[str, ...] | list[str] = (),
        requires: set[str] | tuple[str, ...] | list[str] = (),
        on_change: OnChange | None = None,
    ) -> None:
        """注册一个组件并声明其 produces / requires 依赖规范."""
        comp = CoComponent(
            id=component_id,
            provides=set(provides),
            requires=set(requires),
            on_change=on_change,
        )
        old = self._components.get(component_id)
        if old is not None:
            self._unregister_links(component_id)
        self._components[component_id] = comp
        for key in comp.requires:
            self._dependents[key].add(component_id)
        # 初始激活态静默评估 (不通知, 视作组件就绪时的基线).
        comp.active = comp.evaluate(lambda k: self._available.get(k, False))
        # 上线 provides → 刷新产生键的可用性, 依赖者因此被激活.
        for key in comp.provides:
            self._providers[key].add(component_id)
            self._refresh_key(key)

    def unbind(self, component_id: str) -> None:
        """移除一个组件 (空间卸载). 其提供的 key 若再无活跃 provider,
        级联通知依赖者停用."""
        comp = self._components.get(component_id)
        if comp is None:
            return
        self._unregister_links(component_id)
        self._components.pop(component_id, None)
        for key in comp.provides:
            self._refresh_key(key)

    def _unregister_links(self, component_id: str) -> None:
        comp = self._components.get(component_id)
        if comp is None:
            return
        for key in comp.provides:
            self._providers[key].discard(component_id)
        for key in comp.requires:
            self._dependents[key].discard(component_id)

    # ── 上下文变化驱动 ────────────────────────────────────────────
    def set_available(self, key: str, available: bool) -> None:
        """声明某个 produce 的可用性变化, 触发依赖者激活/停用 + 级联."""
        self._available[key] = available
        for dep_id in tuple(self._dependents.get(key, ())):
            self._recompute(dep_id)

    def _refresh_key(self, key: str) -> None:
        """重算一个 produce 键的可用性 (是否存在活跃 provider), 变化则级联."""
        active_providers = {
            cid
            for cid in self._providers.get(key, ())
            if self._components.get(cid) is not None and self._components[cid].active
        }
        now = bool(active_providers)
        if self._available.get(key) == now:
            return
        self._available[key] = now
        for dep_id in tuple(self._dependents.get(key, ())):
            self._recompute(dep_id)

    def _recompute(self, component_id: str) -> None:
        """重算组件激活态; 变化则通知 on_change, 并级联其 produces."""
        comp = self._components.get(component_id)
        if comp is None:
            return
        new_active = comp.evaluate(lambda k: self._available.get(k, False))
        if new_active == comp.active:
            return
        comp.active = new_active
        action = ACTIVATE if new_active else DEACTIVATE
        if comp.on_change is not None:
            for key in comp.requires:
                try:
                    comp.on_change(key, action)
                except Exception:
                    logger.warning(
                        "coeffect: on_change failed for %s on %s", component_id, key,
                        exc_info=True,
                    )
        # 组件激活态变化 → 其 produces 的可用性随之变化 → 级联.
        for key in comp.provides:
            self._refresh_key(key)

    # ── 查询 ─────────────────────────────────────────────────────
    def providers(self, key: str) -> set[str]:
        return set(self._providers.get(key, ()))

    def dependents(self, key: str) -> set[str]:
        return set(self._dependents.get(key, ()))

    def is_available(self, key: str) -> bool:
        return bool(self._available.get(key))

    def is_active(self, component_id: str) -> bool:
        comp = self._components.get(component_id)
        return bool(comp and comp.active)

    def component(self, component_id: str) -> CoComponent | None:
        return self._components.get(component_id)

    def active_dependencies(self) -> dict[str, bool]:
        """当前所有 produce 键的可用性快照."""
        return dict(self._available)
