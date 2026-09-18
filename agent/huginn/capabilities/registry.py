"""CapabilityRegistry —— 插件能力维度的注册 / 查询 / 卸载 (可逆 mount).

对标 Cordis 的插件能力注册, 但**平行**于既有 StarHandlerRegistry 与 ToolRegistry:
  - 不侵入它们内部;
  - key = (dimension, name); 支持按 plugin 整组卸载 (可逆).
"""
from __future__ import annotations

import threading
from collections.abc import Iterable

from huginn.capabilities.capability import DIMENSIONS, CapabilityMetadata


class CapabilityRegistry:
    """能力维度注册表 (线程安全)."""

    # (dimension, name) -> CapabilityMetadata
    _caps: dict[tuple[str, str], CapabilityMetadata] = {}
    _lock = threading.RLock()

    @classmethod
    def register(cls, *metas: CapabilityMetadata) -> None:
        with cls._lock:
            for m in metas:
                if m.dimension not in DIMENSIONS:
                    continue
                cls._caps[(m.dimension, m.name)] = m

    @classmethod
    def register_iterable(cls, metas: Iterable[CapabilityMetadata]) -> None:
        for m in list(metas):
            cls.register(m)

    @classmethod
    def unregister_plugin(cls, plugin_name: str) -> int:
        """卸载某插件提供的所有能力 (可逆 mount 的逆操作)."""
        removed = 0
        with cls._lock:
            for key in [k for k, m in cls._caps.items() if m.plugin_name == plugin_name]:
                del cls._caps[key]
                removed += 1
        return removed

    @classmethod
    def get(cls, dimension: str, name: str) -> CapabilityMetadata | None:
        with cls._lock:
            return cls._caps.get((dimension, name))

    @classmethod
    def list(cls, dimension: str | None = None) -> list[CapabilityMetadata]:
        with cls._lock:
            items = list(cls._caps.values())
        if dimension is not None:
            items = [m for m in items if m.dimension == dimension]
        return sorted(items, key=lambda m: (m.dimension, m.name))

    @classmethod
    def list_names(cls, dimension: str | None = None) -> list[tuple[str, str]]:
        return [(m.dimension, m.name) for m in cls.list(dimension)]

    @classmethod
    def clear(cls) -> None:
        with cls._lock:
            cls._caps.clear()


# 进程级共享单例 — 与 StarHandlerRegistry.get_shared_registry 同款理由:
# 避免 loader 注册到 A、消费者从 B 查, 永远碰不上.
_shared: CapabilityRegistry | None = None
_shared_lock = threading.Lock()


def get_shared_capability_registry() -> CapabilityRegistry:
    global _shared
    if _shared is None:
        with _shared_lock:
            if _shared is None:
                _shared = CapabilityRegistry()
    return _shared


__all__ = ["CapabilityRegistry", "get_shared_capability_registry"]
