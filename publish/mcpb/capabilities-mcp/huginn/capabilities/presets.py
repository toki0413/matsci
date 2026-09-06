"""组合能力预设 — 用"集装箱"把原子能力编排成流程.

示例: literature_research_capability 把文献检索 → 数值校验 → 综述串成一个
一键调用的复合能力。子能力 AtomCapability (literature_tool 的不同 action)
负责各自动作, CompositeCapability 负责顺序编排与状态传递。

注册: ``register_capability_presets()`` 会把它注册进 CapabilityRegistry,
上层 (LLM / 编排器) 就多了一个"整箱搬运"的能力, 而不是在 132 个碎片工具里
现拼流程。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from huginn.capabilities.base import CapabilityResult
from huginn.capabilities.intents import (
    AtomicCapability,
    CompositeCapability,
)
from huginn.capabilities.registry import CapabilityRegistry
from huginn.tools.registry import ToolRegistry


class LiteratureResearchInput(BaseModel):
    """combination 能力入口: 一次给 query + system/property, 内部拆action跑。"""

    query: str = Field(..., description="文献检索 query")
    system: str | None = Field(default=None, description="体系名 (benchmark_lookup 用)")
    property: str | None = Field(default=None, description="性质名 (benchmark_lookup 用)")
    max_results: int = Field(default=8, ge=1, le=30)
    min_citations: int | None = Field(default=None, description="引用数下限 (L1b)")
    oa_only: bool = Field(default=False, description="只保留开放获取 (L1b)")


def _literature_tool():
    """从注册表取 literature_tool。未注册返回 None (能力不可用)。"""
    return ToolRegistry.get("literature_tool")


def _make_lit_cap(action: str) -> AtomicCapability | None:
    tool = _literature_tool()
    if tool is None:
        return None

    class _LitActionCap(AtomicCapability):
        def __init__(self, action: str):
            super().__init__(tool)
            self.name = f"literature_{action}_cap"
            self.description = f"literature_tool {action} 动作的能力封装"
            self._lit_action = action

        async def _run(self, args, context):
            inner = args
            if isinstance(args, BaseModel):
                inner = args.model_dump()
            call_args = dict(inner)
            call_args["action"] = self._lit_action
            result = await self.tool.call(
                type(self.tool.input_schema)(**call_args), context
            )
            if result.success:
                return CapabilityResult.ok(result.data)
            return CapabilityResult.fail(
                result.error or f"{self._lit_action} failed", code="tool_error"
            )

    return _LitActionCap(action)


def register_capability_presets() -> list[str]:
    """注册内置组合能力预设。返回注册的能力名列表。"""
    registered: list[str] = []

    search_cap = _make_lit_cap("search")
    lookup_cap = _make_lit_cap("benchmark_lookup")
    summarize_cap = _make_lit_cap("summarize")

    if search_cap is None or lookup_cap is None or summarize_cap is None:
        # literature_tool 未注册时, 组合能力没法构建 — 安静跳过
        return registered

    # 步骤转换: 把文献研究入参分流给各 action
    def _to_search(prev, inp: LiteratureResearchInput):
        return {
            "query": inp.query,
            "max_results": inp.max_results,
            "min_citations": inp.min_citations,
            "oa_only": inp.oa_only,
        }

    def _to_lookup(prev, inp: LiteratureResearchInput):
        # 从上一轮 search 结果里取 papers, 喂给 benchmark_lookup 做数值校验
        search_state = prev.get("search")
        papers = []
        if search_state is not None and search_state.success:
            papers = (search_state.data or {}).get("papers", [])[:10]
        return {
            "query": inp.query,
            "system": inp.system or inp.query,
            "property": inp.property or "",
            "papers": papers,
        }

    def _to_summarize(prev, inp: LiteratureResearchInput):
        search_state = prev.get("search")
        papers = []
        if search_state is not None and search_state.success:
            papers = (search_state.data or {}).get("papers", [])[:10]
        return {
            "query": inp.query,
            "papers": papers,
            "focus": f"{inp.system or ''} {inp.property or ''}".strip() or None,
        }

    composite = CompositeCapability(
        name="literature_research_cap",
        description=(
            "一键文献调研: 搜索(OA/引用数过滤) → 跨源数值校验 → 综述. "
            "整箱搬运原子能力, 无需上层现拼流程."
        ),
        category="literature",
        input_schema=LiteratureResearchInput,
        steps=[
            ("search", search_cap, _to_search),
            ("verify", lookup_cap, _to_lookup),
            ("review", summarize_cap, _to_summarize),
        ],
        best_effort={"verify"},  # 数值校验失败不影响综述
        degradation_chain=("web_search_cap",),
    )
    CapabilityRegistry.register(composite)
    registered.append(composite.name)
    return registered
