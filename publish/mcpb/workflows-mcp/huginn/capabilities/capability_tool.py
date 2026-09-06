"""capability_tool — 让 LLM "按能力" 检索和执行, 而非在 132 个碎片工具里找.

这是能力集装箱化与既有 LLM 工具装配链的接缝:
- :class:`CapabilityTool` 是一个普通 HuginnTool, 注册进 ToolRegistry (因而进入
  ``ToolRegistry.list_tools()`` 的装配路径, LLM 始终可见).
- 其动作:
    - ``list``: 返回统一能力清单 (manifest) — name/category/available/
      sub_capabilities/degradation_chain. 让 LLM 一层抵达"整箱能力"而非碎片.
    - ``run``: 按能力名执行 (经 CapabilityRegistry 分派), 支持组合/原子/外部.
    - ``search``: 按关键词过滤能力清单 (轻量能力检索的补充).
- 与 ``tool_search`` (语义搜碎片工具) 互补: 一个搜能力容器, 一个搜原子工具.

存量原子工具仍是工具, 组合/外部能力经这里暴露 — 不重复注册, 避免 schema 爆炸.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.core_types import ToolContext, ToolResult
from huginn.tools.base import HuginnTool
from huginn.tools.profile import ToolProfile


class CapabilityToolInput(BaseModel):
    action: Literal["list", "run", "search"] = Field(
        default="list",
        description="list=返回能力清单; run=按 name 执行能力; "
                    "search=按关键词过滤清单",
    )
    name: str | None = Field(
        default=None,
        description="run 时用: 要执行的能力名 (来自 list 输出)",
    )
    args: dict[str, Any] = Field(
        default_factory=dict,
        description="run 时用: 传给该能力的参数 (dict)",
    )
    query: str = Field(
        default="",
        description="search 时用: 按 name/category/description 关键词过滤",
    )


class CapabilityTool(HuginnTool):
    """按能力检索 + 执行 — 能力集装箱化的 LLM 入口."""

    name = "capability_tool"
    category = "meta"
    description = (
        "List and run composite/atomic capabilities by name, rather than hunting "
        "through dozens of low-level tools. action='list' returns the capability "
        "manifest (with availability and dependencies); action='run' executes a "
        "registered capability; action='search' filters the manifest by keyword. "
        "Prefer this for multi-step tasks (e.g. literature research) that existing "
        "composite capabilities already wrap."
    )
    read_only = False
    input_schema = CapabilityToolInput
    profile = ToolProfile(cost_tier="none", phases=None)

    async def call(self, args: CapabilityToolInput, context: ToolContext | None = None) -> ToolResult:
        from huginn.capabilities.registry import CapabilityRegistry

        if args.action == "list":
            manifest = CapabilityRegistry.manifest()
            return ToolResult(
                data={
                    "action": "list",
                    "n_capabilities": len(manifest),
                    "capabilities": manifest,
                },
                success=True,
            )
        if args.action == "search":
            q = (args.query or "").lower().strip()
            manifest = CapabilityRegistry.manifest()
            if not q:
                return ToolResult(
                    data={
                        "action": "search",
                        "query": q,
                        "n_capabilities": len(manifest),
                        "capabilities": manifest,
                    },
                    success=True,
                )
            hit = [
                m for m in manifest
                if q in (m.get("name", "")).lower()
                or q in (m.get("category", "")).lower()
                or q in (m.get("description", "")).lower()
            ]
            return ToolResult(
                data={
                    "action": "search",
                    "query": q,
                    "n_capabilities": len(hit),
                    "capabilities": hit,
                },
                success=True,
            )
        # run
        name = (args.name or "").strip()
        if not name:
            return ToolResult(
                data=None, success=False,
                error="run action requires 'name' (from list output)",
            )
        result = await CapabilityRegistry.run(name, args.args, self._make_context(context))
        tool_result = result.to_tool_result()
        if not result.success:
            return tool_result
        # run 成功时, 附带能力信息方便 LLM 定位
        tool_result.data = {
            "capability": name,
            "result": result.data,
            **result.metadata,
        }
        return tool_result

    @staticmethod
    def _make_context(context: ToolContext | None) -> ToolContext:
        if context is not None:
            return context
        return ToolContext(session_id="capability_tool", workspace=".")
