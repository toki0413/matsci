"""recall_context tool — G19 外置化上下文按需召回.

agent 在需要时主动 recall 已外置化的 context 段 (autoloop/bench/skill 历史),
不靠 prompt 静态塞进上下文窗口.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from huginn import metacog
from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult


def recall_context(category: str, query: str = "", top_k: int = 3) -> dict:
    """tool 入口: 调 metacog.recall_context, 包装成结构化返回.

    返回:
        {"category": str, "results": list[dict], "count": int}
    """
    results = metacog.recall_context(category=category, query=query, top_k=top_k)
    return {
        "category": category,
        "results": results,
        "count": len(results),
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
        "Recall externalized context segments (autoloop/bench/skill history) on demand. "
        "Use when you need prior autoloop_summary, benchmark_run_summary, "
        "skill_invocation, knowledge_seed, stable_principles, hypothesis, "
        "failure, or subgoal records."
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
