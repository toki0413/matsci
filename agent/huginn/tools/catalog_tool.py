"""catalog_search — 统一目录的 LLM 入口 (只读).

解决"能力孤岛": tool / capability / skill / workflow 各自一套注册表与入口,
LLM 必须猜到该用哪个 meta 工具。本工具用一次检索跨四类返回 name + description
+ 调用提示, 并给出各类计数 —— 某类为 0 时调用方立刻知道"这一类整体缺失".

只读: 不注册、不启停、不执行任何底层条目。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from huginn.core_types import ToolContext, ToolResult
from huginn.tools.base import HuginnTool
from huginn.tools.profile import ToolProfile

_CALL_HINT = {
    "tool": "call this tool directly by name",
    "capability": "capability_tool action='run' with name=<name>",
    "skill": "skill action='execute' with skill_name=<name>",
    "workflow": "workflow_tool action='run' with name=<name> and params={...}",
}


class CatalogSearchInput(BaseModel):
    action: Literal["search", "list"] = Field(
        default="list",
        description="list=列出各 kind 前若干条; search=按关键词检索",
    )
    query: str = Field(default="", description="search 用: 名称/描述/分类关键词")
    kind: str | None = Field(
        default=None,
        description="可选: 限定种类, 支持逗号分隔 (tool,capability,skill,workflow)",
    )
    limit: int = Field(default=20, ge=1, le=100, description="返回条数上限")


class CatalogSearchTool(HuginnTool):
    """跨注册表的统一只读检索."""

    name = "catalog_search"
    category = "meta"
    description = (
        "One read-only search across tools, capabilities, skills and workflows. "
        "Use it when unsure which registry holds a capability, or to check whether "
        "a workflow/skill exists before hand-rolling it. Returns name, description, "
        "and how to invoke each item. action='list' shows counts per kind."
    )
    read_only = True
    destructive = False
    input_schema = CatalogSearchInput
    profile = ToolProfile(cost_tier="none", phases=None)

    async def call(self, args: CatalogSearchInput,
                   context: ToolContext | None = None) -> ToolResult:
        from huginn.catalog.search import search_entries, summarize

        entries = search_entries(
            query=args.query if args.action == "search" else "",
            kind=args.kind,
            limit=args.limit,
        )
        items = [
            {
                "kind": e.kind,
                "name": e.name,
                "description": str((e.meta or {}).get("description", "")),
                "category": str((e.meta or {}).get("category", "")),
                "origin": e.origin,
                "enabled": e.enabled,
                "call_hint": _CALL_HINT.get(e.kind, "see the catalog route"),
            }
            for e in entries
        ]
        return ToolResult(
            data={
                "action": args.action,
                "query": args.query,
                "kind": args.kind,
                "counts": summarize(search_entries(kind=args.kind, limit=10_000)),
                "n_items": len(items),
                "items": items,
            },
            success=True,
        )
