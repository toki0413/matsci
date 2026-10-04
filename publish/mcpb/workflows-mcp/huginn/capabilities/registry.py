"""能力堆场 (Capability registry) — "集装箱"的统一发现/装配/视图.

职责:
1. 注册能力 (原子 / 组合 / 外部 / 显式传入或从 HuginnTool 自动装箱)
2. 自动发现: 从既有 ToolRegistry 加载所有激活工具为原子能力
3. 统一视图: 输出能力清单 (含 sub_capabilities / degradation_chain), 供 LLM
   或上层编排器检索"想要哪个能力", 而非在 132 个碎片工具里迷失
4. 保持可选: 不强制所有工具装箱 — 原子能力按需自动创建, 组合能力显式注册

注: 本模块同时承载两条平行能力体系 (互不覆盖, 各自独立命名, 见下):
  - ``CapabilityRegistry``        : 能力集装箱堆场 (既有, 供编排器/MCP 码头/CLI 使用)
  - ``CapabilityMountRegistry``   : 能力维度 mount 注册表 (P1, 供 ``@capability`` 能力
    维度声明使用) —— 新系统不再占用旧 ``CapabilityRegistry`` 的 API, 避免破坏既有契约.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from typing import Any

from huginn.capabilities.base import Capability, CapabilityResult
from huginn.capabilities.intents import AtomicCapability
from huginn.core_types import ToolContext
from huginn.tools.registry import ToolRegistry

# 下面是两条平行体系的分界.
# 1) 旧: 能力集装箱 CapabilityRegistry (既有接口, 保持 master 契约).
# 2) 新: 能力维度 mount CapabilityMountRegistry (P1, 独立命名, 不覆盖旧类).
#    两个都把"能力维度声明"装在专门类里, 语义清晰互不混淆.


class CapabilityRegistry:
    """能力寄存器。静态方法为主, 与 ToolRegistry 风格一致。

    这是"能力规模 (集装箱)"体系: 每个能力是带 run() 契约的可执行单元
    (原子/组合/外部), 按 name 注册, 支持统一视图与按名调用.
    """

    _capabilities: dict[str, Capability] = {}
    _lock = threading.Lock()
    _auto_atomic: dict[str, AtomicCapability] = {}
    _scanned: bool = False

    # ── 注册 / 移除 ──────────────────────────────────────────────

    @classmethod
    def register(cls, cap: Capability) -> Capability:
        """注册一个能力 (组合 / 外部 / 显式原子)。"""
        if not cap.name:
            raise ValueError("Capability must have a name")
        with cls._lock:
            cls._capabilities[cap.name] = cap
        return cap

    @classmethod
    def unregister(cls, name: str) -> bool:
        with cls._lock:
            cls._capabilities.pop(name, None)
            cls._auto_atomic.pop(name, None)
        return True

    @classmethod
    def clear(cls) -> None:
        with cls._lock:
            cls._capabilities.clear()
            cls._auto_atomic.clear()
            cls._scanned = False

    # ── 自动发现 (从 ToolRegistry 装箱) ─────────────────────────

    @classmethod
    def scan_tool_registry(cls, names: list[str] | None = None) -> list[str]:
        """把所有激活的 HuginnTool 自动装箱为原子能力。

        ``names`` 为 None 时扫描注册表里全部激活工具; 否则只扫描给定子集。
        返回新装箱的能力名列表。
        """
        with cls._lock:
            if cls._scanned and names is None:
                return list(cls._auto_atomic.keys())
            cls._scanned = True
        added: list[str] = []
        for name in ToolRegistry.list_tools():
            if names is not None and name not in names:
                continue
            if name in cls._capabilities or name in cls._auto_atomic:
                continue
            tool = ToolRegistry.get(name)
            if tool is None or not tool.active:
                continue
            cap = AtomicCapability(tool)
            cls._auto_atomic[name] = cap
            added.append(name)
        return added

    # ── 查询 ────────────────────────────────────────────────────

    @classmethod
    def get(cls, name: str) -> Capability | None:
        """按名取能力。显式注册的优先, 其次自动装箱的原子能力, 最后兜底现取现装。"""
        with cls._lock:
            cap = cls._capabilities.get(name)
            if cap is not None:
                return cap
            cap = cls._auto_atomic.get(name)
            if cap is not None:
                return cap
            # 兜底: 若是注册表里的工具名, 现场装箱
            if cls._is_registry_tool(name):
                return AtomicCapability(ToolRegistry.get(name))  # type: ignore[arg-type]
        return None

    @classmethod
    def _is_registry_tool(cls, name: str) -> bool:
        tool = ToolRegistry.get(name)
        return tool is not None and tool.active

    @classmethod
    def list_capabilities(cls) -> list[str]:
        cls.scan_tool_registry()
        with cls._lock:
            return list(dict.fromkeys(list(cls._capabilities.keys()) + list(cls._auto_atomic.keys())))

    def all_raw(self) -> list[Capability]:
        """源码可用, 单测友好: 返回全部 (显式 + 自动) 能力实例。"""
        self.scan_tool_registry()
        with self._lock:
            return list(self._capabilities.values()) + list(self._auto_atomic.values())

    # ── 统一视图 (集装箱清单) ───────────────────────────────────

    @classmethod
    def manifest(cls) -> list[dict[str, Any]]:
        """输出统一能力清单 — 每个能力带 name/category/可用性/依赖/降级链。

        这是"能力清单"层的关键输出: 上层 (LLM / 编排器 / UI) 可按能力而不是
        按碎片工具来检索和编排。
        """
        known: dict[str, Capability] = {}
        with cls._lock:
            known.update(cls._auto_atomic)
            known.update(cls._capabilities)
        out: list[dict[str, Any]] = []
        for name in sorted(known):
            cap = known[name]
            out.append({
                "name": cap.name,
                "category": cap.category,
                "description": cap.description,
                "available": cap.is_available(),
                "read_only": cap.read_only,
                "sub_capabilities": list(cap.sub_capabilities),
                "degradation_chain": list(cap.degradation_chain),
                "input_schema": cap.input_json_schema,
            })
        return out

    # ── 快捷调用 ────────────────────────────────────────────────

    @classmethod
    async def run(cls, name: str, args: Any, context: ToolContext | None = None) -> CapabilityResult:
        """按名执行能力。能力不存在返回失败结果而非抛异常。"""
        cap = cls.get(name)
        if cap is None:
            return CapabilityResult.fail(f"capability not found: {name}", code="not_found")
        if not cap.is_available():
            return CapabilityResult.fail(f"capability unavailable: {name}", code="unavailable")
        return await cap.run(args, context)


def register_capability(cap: Capability) -> Capability:
    """装饰器风格注册。"""
    return CapabilityRegistry.register(cap)


# ────────────────────────────────────────────────────────────────────────
# P1: 能力维度 mount 注册表 —— 独立类, 不覆盖上述 CapabilityRegistry.
#   对标 Corside 的"一切皆插件", 但语义是**维度声明**: key = (dimension, name),
#   支持按 plugin 整组卸载 (可逆 mount). 由 @capability 装饰的插件经
#   loader 挂载; builtin / fusion 也走这里.
# ────────────────────────────────────────────────────────────────────────

from huginn.capabilities.capability import DIMENSIONS, CapabilityMetadata  # noqa: E402


class CapabilityMountRegistry:
    """维度 mount 注册表 (线程安全).

    与 CapabilityRegistry (集装箱堆场) **平行**且独立:
      - CapabilityRegistry 管"可执行能力单元"(run/manifest/scan_tool_registry);
      - CapabilityMountRegistry 管"能力维度声明"(loop/session/storage/sysprompt/fusion),
        按 (dimension, name) 组织, 供 AgentSession/loader 装配.
    """

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


# 进程级共享单例 — 与 CapabilityRegistry 共享同理: loader 注册到 A、
# AgentSession 从 B 查会碰不上, 故都走同一共享实例.
_shared_mount: CapabilityMountRegistry | None = None
_shared_lock = threading.Lock()


def get_shared_capability_registry() -> CapabilityMountRegistry:
    global _shared_mount
    if _shared_mount is None:
        with _shared_lock:
            if _shared_mount is None:
                _shared_mount = CapabilityMountRegistry()
    return _shared_mount


__all__ = [
    "CapabilityRegistry",
    "CapabilityMountRegistry",
    "get_shared_capability_registry",
    "register_capability",
]
