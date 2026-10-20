"""PluginHost — Cordis 式统一挂载点 (时空可组合的最小落地).

Huginn 早已有三个原语, 但各管一段, 互不认识:

- 空间可组合 (CoEffectRegistry): 组件声明 provides/requires, 依赖变化时自动激活/停用;
- 时间可组合 (RevertibleContext): 副作用携带逆, 卸载即恢复;
- 事件订阅 (StarHandlerRegistry + EventBus): handler 注册即订阅.

Cordis 的关键不是"再加一个容器", 而是让这三件事**同生同灭**:

    mount(p)   = 声明依赖 + 订阅 handler + 一切副作用进逆栈
    unmount(p) = 回滚该插件的逆 + 撤 handler + 注销依赖 (级联停用依赖方)

本模块只做编排: 三个原语各自不动 (与 ``security/workspace.py`` 同构 —— 那是物理域
实例, 这是插件域实例). 每个 :class:`Mount` 持有**自己**的 RevertibleContext, 因此
任意顺序卸载都不会误伤别人的逆 (``revert_all`` 只能整体回滚, 故按插件隔离).

不做的: 热模块替换 (HMR) / 配置调和 —— Huginn 无该需求, 不引入.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from huginn.api.event import EventType
from huginn.api.filter import StarHandlerMetadata
from huginn.plugins.event_bus import EventBus
from huginn.plugins.registry import StarHandlerRegistry, get_shared_registry
from huginn.security.coeffect import CoEffectRegistry
from huginn.security.revertible import RevertibleContext

logger = logging.getLogger(__name__)

Disposer = Callable[[], None]


class Mount:
    """一个已挂载插件: 它自己的时间上下文 + 它在共享空间/事件面登记的句柄.

    插件作者只跟这个对象打交道 —— 它就是 Cordis 里"属于该插件的 ctx". 所有
    副作用经 :meth:`effect` / :meth:`track` / :meth:`on` 登记, 卸载时被统一回滚.
    """

    def __init__(
        self,
        host: PluginHost,
        plugin_id: str,
    ) -> None:
        self._host = host
        self.id = plugin_id
        # 每插件独立逆栈 — 保证卸载隔离 (不碰别人的逆).
        self.revertible = RevertibleContext()
        self._unmounted = False

    # ── 时间可组合: 登记副作用 ───────────────────────────────────
    def effect(
        self, op: Callable[[RevertibleContext], tuple[Any, Disposer | None]]
    ) -> Any:
        """执行一个可逆操作并累计其逆 (与 ``RevertibleContext.effect`` 同签名)."""
        return self.revertible.effect(op)

    def track(self, dispose: Disposer | None) -> None:
        """登记一个逆 (卸载时 LIFO 执行)."""
        self.revertible.track(dispose)

    # ── 事件订阅 (注册即效应) ────────────────────────────────────
    def on(
        self,
        event_type: EventType,
        handler: Callable[..., Any],
        *,
        priority: int = 0,
        matchers: list[Callable[[Any], bool]] | None = None,
        permissions: list[str] | None = None,
        name: str = "",
    ) -> Disposer:
        """订阅一个事件: 注册 handler, 并返回"仅撤销这一条订阅"的逆.

        对应 Cordis 的 ``ctx.on(evt, cb) -> disposer``: 订阅本身是一个可逆效应,
        卸载插件时随逆栈一起撤销.
        """
        meta = StarHandlerMetadata(
            handler=handler,
            event_type=event_type,
            plugin_name=self.id,
            priority=priority,
            matchers=list(matchers or []),
            permissions=list(permissions or []),
            name=name or getattr(handler, "__name__", ""),
        )
        self._host.registry.register(meta)

        def dispose() -> None:
            self._host.registry.unregister(meta)

        self.revertible.track(dispose)
        return dispose

    # ── 空间可组合: 就绪度变化 ───────────────────────────────────
    def set_available(self, key: str, available: bool) -> None:
        """声明某个 produce 键的可用性变化 (驱动依赖者激活/停用)."""
        self._host.effects.set_available(key, available)

    @property
    def active(self) -> bool:
        return self._host.effects.is_active(self.id)

    @property
    def unmounted(self) -> bool:
        return self._unmounted

    # ── 卸载 ─────────────────────────────────────────────────────
    def unmount(self) -> None:
        """卸载本插件 (等价于 ``host.unmount(self.id)``). 幂等."""
        self._host.unmount(self.id)


class PluginHost:
    """统一挂载点: 一个 host = 一张空间依赖图 + 一个事件注册表 + 事件总线.

    三个面共享同一份 registry / effects, 插件的挂载与卸载同时作用于三者.
    """

    def __init__(
        self,
        *,
        registry: StarHandlerRegistry | None = None,
        bus: EventBus | None = None,
        effects: CoEffectRegistry | None = None,
    ) -> None:
        self.registry = registry if registry is not None else get_shared_registry()
        self.effects = effects if effects is not None else CoEffectRegistry()
        self.bus = bus if bus is not None else EventBus(registry=self.registry)
        self._mounts: dict[str, Mount] = {}

    # ── 挂载 ─────────────────────────────────────────────────────
    def mount(
        self,
        plugin_id: str,
        *,
        provides: set[str] | tuple[str, ...] | list[str] = (),
        requires: set[str] | tuple[str, ...] | list[str] = (),
        on_dependency_change: Callable[[str, str], None] | None = None,
    ) -> Mount:
        """挂载一个插件: 声明 provides/requires, 返回该插件的 :class:`Mount`.

        ``on_dependency_change(key, action)`` 在依赖满足/消失时被回调
        (action ∈ {"activate", "deactivate"}); 依赖缺失时组件为未激活态,
        其 provides 对其他组件不可见, 因此级联停用下游.
        """
        if plugin_id in self._mounts:
            raise ValueError(f"plugin {plugin_id!r} already mounted")
        self.effects.declare(
            plugin_id,
            provides=provides,
            requires=requires,
            on_change=on_dependency_change,
        )
        m = Mount(self, plugin_id)
        self._mounts[plugin_id] = m
        return m

    # ── 卸载 ─────────────────────────────────────────────────────
    def unmount(self, plugin_id: str) -> None:
        """卸载插件, 三面同退 (幂等, 未挂载则为空操作):

        1. **时间**: 回滚该插件自己的逆 (含撤销其全部 handler 订阅), LIFO;
        2. **事件**: 兜底清掉该插件名下所有 handler (防漏网);
        3. **空间**: 注销依赖声明 → 级联停用依赖它的组件.
        """
        m = self._mounts.pop(plugin_id, None)
        if m is None:
            return
        m._unmounted = True
        # 1) 时间: 该插件自己的逆 (含订阅撤销), 不碰其他插件的逆.
        m.revertible.revert_all()
        # 2) 事件: 兜底 — 以防有 handler 未经 on() 登记 (遗留/第三方).
        self.registry.unregister_plugin(plugin_id)
        # 3) 空间: 注销 → 依赖它的组件级联停用.
        self.effects.unbind(plugin_id)

    # ── 查询 ─────────────────────────────────────────────────────
    def mounted(self) -> list[str]:
        return sorted(self._mounts)

    def get(self, plugin_id: str) -> Mount | None:
        return self._mounts.get(plugin_id)

    def is_active(self, plugin_id: str) -> bool:
        return self.effects.is_active(plugin_id)

    def is_available(self, key: str) -> bool:
        return self.effects.is_available(key)

    def set_available(self, key: str, available: bool) -> None:
        self.effects.set_available(key, available)


__all__ = ["Mount", "PluginHost"]
