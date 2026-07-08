"""subagent_tool -- lets the main agent dispatch isolated subagents.

Wraps SubagentDispatch so the agent can offload context-heavy tasks
(explore, code, analyze) to isolated sessions without bloating the main
conversation window. Inspired by Kimi Code's coder/explore/plan pattern.

Actions:
  - list_types: 列出所有可用的子 agent 类型
  - dispatch:   派发一个子 agent 执行任务, 返回压缩后的摘要
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult


class SubagentToolInput(BaseModel):
    action: Literal["dispatch", "list_types"] = Field(
        default="list_types",
        description="dispatch to run a subagent, list_types to see available types",
    )
    spec_name: str | None = Field(
        default=None,
        description="Subagent type to dispatch (e.g. explore, coder, analyst)",
    )
    task: str | None = Field(
        default=None,
        description="Task description for the subagent to execute",
    )
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional context to pass to the subagent (merged with tool context)",
    )


class SubagentToolOutput(BaseModel):
    success: bool
    action: str
    summary: str | None = None
    tool_calls: list[dict] | None = None
    tokens_used: int | None = None
    available_types: list[dict] | None = None
    error: str | None = None


class SubagentTool(HuginnTool[SubagentToolInput, SubagentToolOutput]):
    """Bridge between the main agent and the subagent dispatch system."""

    name = "subagent_tool"
    category = "meta"
    description = (
        "Dispatch isolated subagents to handle context-heavy tasks. "
        "Each subagent runs in its own session with a restricted tool set. "
        "Use 'list_types' to see available subagent types, 'dispatch' to run one. "
        "Types: explore (read-only search), coder (write/modify code), "
        "analyst (analyze data/results)."
    )
    destructive = False
    read_only = False  # coder subagent can modify files
    input_schema = SubagentToolInput
    output_schema = SubagentToolOutput

    def __init__(self) -> None:
        super().__init__()
        # 延迟导入避免 agents -> tools 循环依赖
        from huginn.agents.subagent import SubagentDispatch

        self._dispatch = SubagentDispatch()

    async def _execute(
        self, args: SubagentToolInput, context: ToolContext
    ) -> ToolResult:
        if args.action == "list_types":
            return self._list_types()
        if args.action == "dispatch":
            return await self._dispatch_subagent(args, context)

        msg = f"Unknown action: {args.action}"
        return ToolResult(
            data=SubagentToolOutput(
                success=False, action=args.action, error=msg
            ).model_dump(),
            success=False,
            error=msg,
        )

    # -- actions -----------------------------------------------------------

    def _list_types(self) -> ToolResult:
        types = self._dispatch.list_specs()
        out = SubagentToolOutput(
            success=True,
            action="list_types",
            available_types=types,
            summary=f"{len(types)} subagent types available",
        )
        return ToolResult(data=out.model_dump(), success=True)

    async def _dispatch_subagent(
        self, args: SubagentToolInput, context: ToolContext
    ) -> ToolResult:
        if not args.spec_name:
            return self._missing_field("spec_name")
        if not args.task:
            return self._missing_field("task")

        # 把 ToolContext 的字段并进 dispatch context dict
        dispatch_ctx = dict(args.context)
        dispatch_ctx.setdefault("agent_factory", context.agent_factory)
        dispatch_ctx.setdefault("session_id", context.session_id)
        dispatch_ctx.setdefault("workspace", context.workspace)

        result = await self._dispatch.dispatch(
            args.spec_name, args.task, dispatch_ctx
        )

        out = SubagentToolOutput(
            success=result.success,
            action="dispatch",
            summary=result.summary,
            tool_calls=result.tool_calls,
            tokens_used=result.tokens_used,
            error=result.error,
        )
        return ToolResult(
            data=out.model_dump(),
            success=result.success,
            error=result.error,
        )

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _missing_field(field: str) -> ToolResult:
        msg = f"{field} is required for dispatch action"
        return ToolResult(
            data=SubagentToolOutput(
                success=False, action="dispatch", error=msg
            ).model_dump(),
            success=False,
            error=msg,
        )
