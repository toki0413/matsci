"""Tool registry — inspired by Claude Code's tools.ts.

Centralized tool registration and discovery.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from huginn.tools.assembly import annotate_metadata, assemble_tool_pool
from huginn.tools.defaults import ToolMetadata

if TYPE_CHECKING:
    from huginn.tools.base import HuginnTool


class ToolRegistry:
    """Registry for all available tools."""

    _tools: dict[str, HuginnTool] = {}

    @classmethod
    def register(cls, tool: HuginnTool) -> HuginnTool:
        """Register a tool instance."""
        if not tool.name:
            raise ValueError("Tool must have a name")
        cls._tools[tool.name] = tool
        return tool

    @classmethod
    def get(cls, name: str) -> HuginnTool | None:
        return cls._tools.get(name)

    @classmethod
    def list_tools(cls) -> list[str]:
        return list(cls._tools.keys())

    @classmethod
    def get_all_schemas(cls) -> list[dict]:
        """Get JSON schemas for all registered tools (for LLM function calling)."""
        schemas = []
        for name, tool in cls._tools.items():
            schema = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.description,
                    "parameters": tool.input_json_schema
                    or {"type": "object", "properties": {}},
                },
                "destructive": tool.destructive,
                "read_only": tool.read_only,
                # fail-closed 默认值: 未显式声明只读的工具一律需要确认
                "metadata": ToolMetadata(
                    is_read_only=tool.read_only,
                    is_destructive=tool.destructive,
                    requires_confirmation=not tool.read_only,
                ),
            }
            schemas.append(schema)
        return schemas

    @classmethod
    def get_assembled_schemas(
        cls,
        permission_rules: dict | None = None,
        mcp_tools: list[dict] | None = None,
    ) -> list[dict]:
        """装配后的工具 schema 列表 —— deny 工具在装配阶段就被过滤。

        单一装配点: 合并内置工具与 MCP 工具 → 过滤 deny → 排序去重 → 标注 metadata。
        这样 LLM 提示词里的工具列表与实际可用工具始终一致, 便于缓存。
        """
        builtin = cls.get_all_schemas()
        assembled = assemble_tool_pool(
            builtin_tools=builtin,
            mcp_tools=mcp_tools,
            permission_rules=permission_rules,
        )
        return annotate_metadata(assembled)

    @classmethod
    def unregister(cls, name: str) -> bool:
        """Remove a tool from the registry."""
        return cls._tools.pop(name, None) is not None

    @classmethod
    def clear(cls) -> None:
        cls._tools.clear()


def register_tool(tool: HuginnTool) -> HuginnTool:
    """Decorator-style registration."""
    return ToolRegistry.register(tool)
