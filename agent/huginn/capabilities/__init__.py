"""能力集装箱化抽象层 (Capability container).

把一个复杂能力封装成"标准集装箱": 统一契约 + 能力清单 + 组合复用 + 外部接入.

- :class:`Capability` — 统一契约基类 (name/category/input/output/is_available/组合/降级)
- :class:`AtomicCapability` — 包装单个 HuginnTool 的原子能力
- :class:`CompositeCapability` — 编排多个子能力为一个流程的组合能力
- :class:`CapabilityRegistry` — 集装箱堆场: 自动发现/装配/统一视图

复用既有工具体系:
- 原子能力直接调 ToolRegistry 里的 HuginnTool
- 组合能力用 DAG 编排子能力, 输出标准化 ToolResult
- 外部能力经 MCPToolAdapter 包装进同一契约
"""

from __future__ import annotations

from huginn.capabilities.base import (
    Capability,
    CapabilityError,
    CapabilityResult,
)
from huginn.capabilities.intents import (
    AtomicCapability,
    CompositeCapability,
    ExternalCapability,
    capability_from_tool,
)
from huginn.capabilities.registry import CapabilityRegistry
from huginn.capabilities.presets import register_capability_presets

__all__ = [
    "Capability",
    "CapabilityError",
    "CapabilityResult",
    "AtomicCapability",
    "CompositeCapability",
    "ExternalCapability",
    "capability_from_tool",
    "CapabilityRegistry",
    "register_capability_presets",
]