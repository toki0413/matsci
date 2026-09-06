"""能力堆场 (Capability registry) — "集装箱"的统一发现/装配/视图.

职责:
1. 注册能力 (原子 / 组合 / 外部 / 显式传入或从 HuginnTool 自动装箱)
2. 自动发现: 从既有 ToolRegistry 加载所有激活工具为原子能力
3. 统一视图: 输出能力清单 (含 sub_capabilities / degradation_chain), 供 LLM
   或上层编排器检索"想要哪个能力", 而非在 132 个碎片工具里迷失
4. 保持可选: 不强制所有工具装箱 — 原子能力按需自动创建, 组合能力显式注册
"""

from __future__ import annotations

import threading
from typing import Any

from huginn.capabilities.base import Capability, CapabilityResult
from huginn.capabilities.intents import AtomicCapability
from huginn.core_types import ToolContext
from huginn.tools.registry import ToolRegistry


class CapabilityRegistry:
    """能力寄存器。静态方法为主, 与 ToolRegistry 风格一致。"""

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
