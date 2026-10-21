"""Agent-facing memory tools: remember and recall.

These allow the LLM to store facts/insights and search its own long-term
memory during a conversation.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from huginn.core_types import ToolResult
from huginn.tools.base import HuginnTool


class RememberInput(BaseModel):
    content: str = Field(description="The fact, insight, or observation to remember.")
    category: str = Field(
        default="fact",
        description="Memory category: fact, insight, conversation, calculation, error, episode.",
    )
    tags: list[str] = Field(default_factory=list, description="Tags for filtering.")
    importance: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Importance score 0-1."
    )
    tier: str = Field(
        default="mid",
        description="Memory tier: short (6h), mid (7d), long (permanent).",
    )


class RememberOutput(BaseModel):
    memory_id: str
    success: bool


class RecallInput(BaseModel):
    query: str = Field(
        default="", description="Search query. Empty is allowed for category/handle recall."
    )
    category: str | None = Field(
        default=None,
        description=(
            "Category filter. Also selects context recall when mode='auto' "
            "(e.g. autoloop_summary / failure / hypothesis)."
        ),
    )
    tier: str | None = Field(
        default=None, description="Optional tier filter: short, mid, long."
    )
    top_k: int = Field(
        default=5, ge=1, le=50, description="Number of records to retrieve."
    )
    mode: str = Field(
        default="auto",
        description=(
            "Recall backend: auto | memory | context | observation. "
            "auto routes by handle > category > semantic memory search."
        ),
    )
    handle: str | None = Field(
        default=None,
        description="Observation handle '@obs:<uuid>' for exact paged recall of offloaded output.",
    )
    page: int = Field(default=0, ge=0, description="Page index for observation recall.")
    page_size: int = Field(
        default=4000, ge=1, le=20000, description="Characters per page for observation recall."
    )


class RecallOutput(BaseModel):
    results: list[dict]
    intent: str = ""
    source: str = ""
    available: bool = True
    count: int = 0
    error: str | None = None


class RememberTool(HuginnTool[RememberInput, RememberOutput]):
    name = "remember"
    category = "meta"
    description = "Store a fact, insight, or observation into long-term memory."
    destructive = False
    read_only = False
    input_schema = RememberInput
    output_schema = RememberOutput

    async def call(self, args: RememberInput, context) -> ToolResult:
        if context.memory_manager is None:
            return ToolResult(
                data=None,
                success=False,
                error="Memory manager not available in this context.",
            )
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
    category = "meta"
    description = (
        "Unified recall entry. Routes one call to the right backend by intent "
        "(mode='auto': handle > category > semantic memory), so you do not need "
        "to pick between memory / externalized context / archived output."
    )
    destructive = False
    read_only = True
    input_schema = RecallInput
    output_schema = RecallOutput

    async def call(self, args: RecallInput, context) -> ToolResult:
        from huginn.tools.recall_facade import route_recall

        outcome = route_recall(
            query=args.query,
            category=args.category,
            top_k=args.top_k,
            tier=args.tier,
            handle=args.handle,
            page=args.page,
            page_size=args.page_size,
            mode=args.mode,
            memory_manager=getattr(context, "memory_manager", None),
            session_id=getattr(context, "session_id", "default"),
        )
        output = RecallOutput(
            results=outcome["results"],
            intent=outcome["intent"],
            source=outcome["source"],
            available=outcome["available"],
            count=outcome["count"],
            error=outcome["error"],
        )
        return ToolResult(
            data=output.model_dump(),
            success=outcome["available"],
            error=outcome["error"],
        )
