"""Tool registry — inspired by Claude Code's tools.ts.

Centralized tool registration and discovery.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from huginn.tools.assembly import annotate_metadata, assemble_tool_pool
from huginn.tools.defaults import ToolMetadata

if TYPE_CHECKING:
    from huginn.tools.base import HuginnTool


class ToolRegistry:
    """Registry for all available tools."""

    _tools: dict[str, HuginnTool] = {}
    _schemas_cache: list[dict] | None = None
    _lock = threading.Lock()

    @classmethod
    def register(cls, tool: HuginnTool) -> HuginnTool:
        """Register a tool instance."""
        if not tool.name:
            raise ValueError("Tool must have a name")
        with cls._lock:
            cls._tools[tool.name] = tool
            cls._schemas_cache = None
        return tool

    @classmethod
    def get(cls, name: str) -> HuginnTool | None:
        with cls._lock:
            return cls._tools.get(name)

    @classmethod
    def list_tools(cls) -> list[str]:
        with cls._lock:
            return list(cls._tools.keys())

    @classmethod
    def get_all_schemas(cls) -> list[dict]:
        """Get JSON schemas for all registered tools (for LLM function calling).

        Filters on two layers:
          - active=False: tool is administratively hidden from the LLM but
            still callable via ToolRegistry.get() directly (e.g. disabled
            via config). Static, cached.
          - is_available()=False: tool's external resource is currently down
            (MCP server disconnected, optional dep missing). Runtime state,
            re-checked on every call because caching it would freeze the
            schema list across reconnects.
        """
        # Tools are registered at startup and don't change at runtime,
        # so we cache the serialized schemas to avoid re-serializing 132
        # tools on every /tools request.
        with cls._lock:
            if cls._schemas_cache is None:
                schemas = []
                for name, tool in cls._tools.items():
                    if not tool.active:
                        continue
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
                        "metadata": ToolMetadata(
                            is_read_only=tool.read_only,
                            is_destructive=tool.destructive,
                            requires_confirmation=not tool.read_only,
                        ),
                    }
                    schemas.append(schema)
                cls._schemas_cache = schemas

            unavailable = [
                name for name, tool in cls._tools.items()
                if not getattr(tool, "is_available", lambda: True)()
            ]
            if not unavailable:
                return cls._schemas_cache
            return [
                s for s in cls._schemas_cache
                if s["function"]["name"] not in unavailable
            ]

    @classmethod
    def get_schemas_for_provider(cls, provider: str = "openai") -> list[dict]:
        """按指定 LLM provider 的格式返回工具 schema。

        借鉴 AstrBot ToolSet 的 openai_schema()/anthropic_schema()/google_schema()
        思路: 同一份工具定义, 按需转成不同 provider 期望的格式。
        会过滤掉 active=False 的工具 (对 LLM 不可见)。
        """
        from huginn.tools.schema_adapters import adapt_schemas

        schemas = cls.get_all_schemas()
        # get_all_schemas 已经过滤了 inactive 工具, 这里再兜一层防御:
        # 万一缓存是工具被禁用前生成的, 也保证不泄漏
        active_schemas = [
            s for s in schemas
            if cls._tools.get(s["function"]["name"]) is None
            or cls._tools[s["function"]["name"]].active
        ]
        return adapt_schemas(active_schemas, provider)

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
        removed = cls._tools.pop(name, None) is not None
        if removed:
            cls._schemas_cache = None
        return removed

    @classmethod
    def clear(cls) -> None:
        cls._tools.clear()
        cls._schemas_cache = None


def register_tool(tool: HuginnTool) -> HuginnTool:
    """Decorator-style registration."""
    return ToolRegistry.register(tool)
