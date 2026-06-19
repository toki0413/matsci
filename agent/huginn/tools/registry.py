"""Tool registry — inspired by Claude Code's tools.ts.

Centralized tool registration and discovery.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
            }
            schemas.append(schema)
        return schemas

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
