"""StarHandlerRegistry —— handler 注册表, 按 priority 降序分发。

借鉴 AstrBot 的 StarHandlerMetadata + priority 排序。
关键点:
  - 按 EventType 索引, dispatch 时 O(1) 找到候选
  - 同 EventType 内按 priority 降序 (越大越先执行)
  - 支持插件级 enable/disable, 禁用后所有 handler 不参与分发
  - 注册时不检查权限, 权限由 PermissionChecker 在 dispatch 时强制
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from huginn.api.event import Event, EventType
from huginn.api.filter import StarHandlerMetadata


@dataclass
class StarHandlerRegistry:
    """handler 注册表。

    线程安全: 内部用 lock 保护 _handlers 和 _disabled。
    实际 dispatch 在 EventBus 里, registry 只负责查询。
    """

    # event_type -> 按 priority 降序的 metadata 列表
    _handlers: dict[EventType, list[StarHandlerMetadata]] = field(
        default_factory=lambda: defaultdict(list)
    )
    # plugin_name -> 是否启用 (默认 True, 注册即启用)
    _disabled: set[str] = field(default_factory=set)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def register(self, *metas: StarHandlerMetadata) -> None:
        """注册一条或多条 metadata。自动按 priority 降序插入。"""
        with self._lock:
            for m in metas:
                lst = self._handlers[m.event_type]
                lst.append(m)
                # 稳定排序: 同 priority 保持注册顺序
                lst.sort(key=lambda x: x.priority, reverse=True)

    def register_iterable(self, metas: Iterable[StarHandlerMetadata]) -> None:
        """便捷方法: register(*list) 的迭代器友好版。"""
        metas = list(metas)
        if metas:
            self.register(*metas)

    def unregister_plugin(self, plugin_name: str) -> int:
        """卸载某插件的所有 handler。返回移除条数。"""
        removed = 0
        with self._lock:
            for evt_type in list(self._handlers.keys()):
                before = len(self._handlers[evt_type])
                self._handlers[evt_type] = [
                    m for m in self._handlers[evt_type] if m.plugin_name != plugin_name
                ]
                removed += before - len(self._handlers[evt_type])
            self._disabled.discard(plugin_name)
        return removed

    def enable(self, plugin_name: str) -> None:
        with self._lock:
            self._disabled.discard(plugin_name)

    def disable(self, plugin_name: str) -> None:
        """禁用插件: handler 不删, 但 dispatch 时跳过。

        跟 unregister 的区别: enable 可恢复, 不需要重新加载。
        """
        with self._lock:
            self._disabled.add(plugin_name)

    def is_enabled(self, plugin_name: str) -> bool:
        with self._lock:
            return plugin_name not in self._disabled

    def get_handlers(self, event_type: EventType) -> list[StarHandlerMetadata]:
        """返回某事件类型的所有启用 handler, 已按 priority 降序。

        返回副本, 调用方修改不影响内部状态。
        """
        with self._lock:
            lst = self._handlers.get(event_type, [])
            return [m for m in lst if m.plugin_name not in self._disabled]

    def get_handlers_for(self, event: Event) -> list[StarHandlerMetadata]:
        """返回能处理该 event 的所有 handler (含 matcher 过滤)。

        顺序: priority 降序。matcher 不通过的剔除。
        """
        candidates = self.get_handlers(event.type)
        return [m for m in candidates if m.matches(event)]

    def list_plugins(self) -> list[str]:
        """列出所有已注册插件名。"""
        with self._lock:
            names: set[str] = set()
            for lst in self._handlers.values():
                for m in lst:
                    names.add(m.plugin_name)
            return sorted(names)

    def clear(self) -> None:
        with self._lock:
            self._handlers.clear()
            self._disabled.clear()

    def __len__(self) -> int:
        with self._lock:
            return sum(len(lst) for lst in self._handlers.values())


# ── 进程级共享 registry ────────────────────────────────────────────
# 解决的问题: engine 和 PluginLoader 各自 new EventBus 时, 每次创建独立
# 的空 registry, 导致 handler 注册到 A, 事件从 B 发出, 永远碰不上.
# 共享单例后, 不管谁 new EventBus 都拿到同一个 registry, handler 不丢.

_shared_registry: StarHandlerRegistry | None = None
_shared_lock = threading.Lock()


def get_shared_registry() -> StarHandlerRegistry:
    """返回进程级共享的 StarHandlerRegistry 单例.

    所有 EventBus 实例默认用这个 registry, 这样 PluginLoader 注册的
    handler 和 engine dispatch 的事件能碰到一起.
    """
    global _shared_registry
    if _shared_registry is None:
        with _shared_lock:
            if _shared_registry is None:
                _shared_registry = StarHandlerRegistry()
    return _shared_registry


__all__ = ["StarHandlerRegistry", "get_shared_registry"]
