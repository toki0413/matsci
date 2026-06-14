"""Agent-facing memory tools: remember and recall.

These allow the LLM to store facts/insights and search its own long-term
memory during a conversation.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool
from huginn.types import ToolResult


class RememberInput(BaseModel):
    content: str = Field(description="The fact, insight, or observation to remember.")
    category: str = Field(default="fact", description="Memory category: fact, insight, conversation, calculation, error, episode.")
    tags: list[str] = Field(default_factory=list, description="Tags for filtering.")
    importance: float = Field(default=0.5, ge=0.0, le=1.0, description="Importance score 0-1.")
    tier: str = Field(default="mid", description="Memory tier: short (6h), mid (7d), long (permanent).")


class RememberOutput(BaseModel):
    memory_id: str
    success: bool


class RecallInput(BaseModel):
    query: str = Field(description="Search query for long-term memory.")
    category: str | None = Field(default=None, description="Optional category filter.")
    tier: str | None = Field(default=None, description="Optional tier filter: short, mid, long.")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of memories to retrieve.")


class RecallOutput(BaseModel):
    results: list[dict]


class RememberTool(HuginnTool[RememberInput, RememberOutput]):
    name = "remember"
    description = "Store a fact, insight, or observation into long-term memory."
    destructive = False
    read_only = False
    input_schema = RememberInput
    output_schema = RememberOutput

    async def call(self, args: RememberInput, context) -> ToolResult:
        if context.memory_manager is None:
            return ToolResult(data=None, success=False, error="Memory manager not available in this context.")
        try:
            mid = context.memory_manager.remember(
                content=args.content,
                category=args.category,
                tags=args.tags,
                importance=args.importance,
                tier=args.tier,
            )
            return ToolResult(
                data=RememberOutput(memory_id=mid, success=True).model_dump(),
                success=True,
                side_effects=[f"Stored memory {mid}"],
            )
        except Exception as exc:
            return ToolResult(data=None, success=False, error=str(exc))


class RecallTool(HuginnTool[RecallInput, RecallOutput]):
    name = "recall"
    description = "Search long-term memory for relevant facts, insights, or prior conversations."
    destructive = False
    read_only = True
    input_schema = RecallInput
    output_schema = RecallOutput

    async def call(self, args: RecallInput, context) -> ToolResult:
        if context.memory_manager is None:
            return ToolResult(data=None, success=False, error="Memory manager not available in this context.")
        try:
            results = context.memory_manager.recall(
                query=args.query,
                category=args.category,
                tier=args.tier,
                top_k=args.top_k,
            )
            return ToolResult(
                data=RecallOutput(results=results).model_dump(),
                success=True,
            )
        except Exception as exc:
            return ToolResult(data=None, success=False, error=str(exc))
