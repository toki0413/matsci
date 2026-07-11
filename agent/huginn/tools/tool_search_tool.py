"""Tool search & invoke — 渐进式工具发现元工具.

借鉴 OpenAaaS 的渐进式能力发现: 不把 60+ 工具 schema 全塞进 context,
而是让 LLM 按需搜索, 找到后直接 proxy-call.
这样能大幅减少 system prompt 的 token 占用.

两个 action:
  - search: 关键词搜索, 返回 name + description + 参数摘要
  - invoke: 按名称调用工具, 代理执行
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult


class ToolSearchInput(BaseModel):
    action: Literal["search", "invoke"] = Field(
        ..., description="search=搜索工具; invoke=代理调用指定工具"
    )
    query: str | None = Field(
        default=None,
        description="search 时用: 关键词, 匹配工具名或描述",
    )
    tool_name: str | None = Field(
        default=None,
        description="invoke 时用: 要调用的工具名",
    )
    tool_args: dict[str, Any] | None = Field(
        default=None,
        description="invoke 时用: 传给目标工具的参数",
    )
    limit: int = Field(
        default=10, ge=1, le=30,
        description="search 时最多返回几条结果",
    )


class ToolSearchTool(HuginnTool):
    """搜索和调用已注册但未在当前 schema 中暴露的工具."""

    name = "tool_search"
    category = "meta"
    description = (
        "Search for and invoke tools not in the current schema. "
        "Use action='search' with a query to find tools by keyword, "
        "then action='invoke' with tool_name and tool_args to call them. "
        "This gives access to the full tool registry without bloating the context."
    )
    read_only = False
    input_schema = ToolSearchInput

    async def call(self, args: ToolSearchInput, context: ToolContext) -> ToolResult:
        from huginn.tools.registry import ToolRegistry

        if args.action == "search":
            return self._search(args.query or "", args.limit)
        if args.action == "invoke":
            return await self._invoke(args.tool_name or "", args.tool_args or {}, context)
        return ToolResult(data=None, success=False, error=f"unknown action: {args.action}")

    def _search(self, query: str, limit: int) -> ToolResult:
        from huginn.tools.registry import ToolRegistry

        q = query.lower().strip()
        results: list[dict[str, Any]] = []
        for name, tool in ToolRegistry._tools.items():
            # 跳过自己, 避免递归
            if name == self.name:
                continue
            desc = tool.description or ""
            # 匹配 name 或 description
            if q and q not in name.lower() and q not in desc.lower():
                continue
            # 参数摘要: 只取字段名和 type, 不展开完整 schema
            params: list[str] = []
            schema = tool.input_json_schema
            if schema and "properties" in schema:
                for pname, pinfo in schema["properties"].items():
                    ptype = pinfo.get("type", "?")
                    params.append(f"{pname}:{ptype}")
            results.append({
                "name": name,
                "description": desc[:200],
                "active": tool.active,
                "params": params[:8],  # 最多列 8 个参数, 避免太长
                "category": getattr(tool, "category", ""),
            })
            if len(results) >= limit:
                break

        return ToolResult(
            data={
                "query": query,
                "count": len(results),
                "tools": results,
            },
            success=True,
        )

    async def _invoke(
        self, tool_name: str, tool_args: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        from huginn.tools.registry import ToolRegistry

        tool = ToolRegistry.get(tool_name)
        if tool is None:
            return ToolResult(
                data=None, success=False,
                error=f"tool '{tool_name}' not found in registry",
            )
        try:
            result = await tool.call(tool_args, context)
            return result
        except Exception as exc:
            return ToolResult(data=None, success=False, error=str(exc))
