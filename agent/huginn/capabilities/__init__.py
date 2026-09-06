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
from huginn.capabilities.presets import register_capability_presets
from huginn.capabilities.registry import CapabilityRegistry

# 对外"MCP 码头" (共享经济/生态融入): 能力 → 标准 MCP server / OpenAI function.
# 经 PEP 562 惰性导出 (见 __getattr__), 保证 `python -m
# huginn.capabilities.mcp_export` 能把该模块当 __main__ 干净加载.
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
    # MCP 码头 (惰性)
    "CapabilityMCPBackend",
    "as_mcp_tools",
    "as_openai_functions",
    "server",
    "list_json",
]


def __getattr__(name: str):
    """PEP 562 惰性暴露 MCP 码头符号.

    不在包顶层 eager import ``mcp_export``: 否则 ``python -m
    huginn.capabilities.mcp_export`` 时该模块会被包先装入 sys.modules,
    ``if __name__ == '__main__'`` 永不触发, `-m` 起不了 server. 惰性导出
    让 ``from huginn.capabilities import server`` 依旧可用, 又不污染包导入.
    """
    if name in {
        "CapabilityMCPBackend",
        "as_mcp_tools",
        "as_openai_functions",
        "server",
        "list_json",
    }:
        from huginn.capabilities import mcp_export as _m

        return getattr(_m, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
