"""recall_context tool — G19 外置化上下文按需召回.

agent 在需要时主动 recall 已外置化的 context 段 (autoloop/bench/skill 历史),
不靠 prompt 静态塞进上下文窗口.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from huginn.core_types import ToolContext, ToolResult
from huginn.tools.base import HuginnTool


def recall_context(category: str, query: str = "", top_k: int = 3) -> dict:
    """tool 入口: 委托统一召回门面 (mode='context'), 包装成结构化返回.

    这是 ``recall`` 工具的类型化窄门 (老调用方按 category 直调). 路由逻辑只有
    ``recall_facade.route_recall`` 一份, 此处不复制后端判定.

    返回:
        {"category": str, "results": list[dict], "count": int, "available": bool}
    """
    from huginn.tools.recall_facade import route_recall

    outcome = route_recall(category=category, query=query, top_k=top_k, mode="context")
    return {
        "category": category,
        "results": outcome["results"],
        "count": outcome["count"],
        "available": outcome["available"],
        "error": outcome["error"],
    }


class RecallContextToolInput(BaseModel):
    category: str = Field(
        ...,
        description=(
            "要 recall 的 context 类别, 如 autoloop_summary / "
            "benchmark_run_summary / skill_invocation / knowledge_seed / "
            "stable_principles / hypothesis / failure / subgoal"
        ),
    )
    query: str = Field(default="", description="可选 FTS 过滤关键词")
    top_k: int = Field(default=3, ge=1, le=50, description="最多返回条数")


class RecallContextTool(HuginnTool):
    """recall_context: 按类别从 long-term memory 召回外置化上下文段."""

    name = "recall_context"
    category = "meta"
    description = (
        "Typed shim for category recall; prefer `recall` with mode='context'. "
        "Recalls externalized context segments (autoloop/bench/skill history) on demand: "
        "autoloop_summary, benchmark_run_summary, skill_invocation, knowledge_seed, "
        "stable_principles, hypothesis, failure, or subgoal records."
    )
    read_only = True
    input_schema = RecallContextToolInput

    def is_read_only(self, args: RecallContextToolInput) -> bool:
        return True

    async def call(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        input_data = RecallContextToolInput(**args)
        try:
            data = recall_context(
                category=input_data.category,
                query=input_data.query,
                top_k=input_data.top_k,
            )
            return ToolResult(data=data, success=True)
        except Exception as e:
            return ToolResult(
                data=None, success=False, error=f"recall_context failed: {e}"
            )


if __name__ == "__main__":
    # self-check: 调一次, 验证返回结构. 空库也该返回 count=0, 不依赖任何预置 memory.
    d = recall_context(category="test_category")
    assert isinstance(d, dict), f"expected dict, got {type(d)}"
    assert "category" in d, f"missing 'category' key: {d}"
    assert "results" in d, f"missing 'results' key: {d}"
    assert "count" in d, f"missing 'count' key: {d}"
    assert d["category"] == "test_category"
    assert d["count"] == len(d["results"])
    print("recall_context self-check PASS")
    print(f"  category={d['category']} count={d['count']}")
