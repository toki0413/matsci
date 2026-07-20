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
    action: Literal["dispatch", "dispatch_parallel", "list_types"] = Field(
        default="list_types",
        description="dispatch to run a subagent, dispatch_parallel for DAG-aware parallel, list_types to see available types",
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
    # v14 Task 13: PersistentTerminal 接入. None 时看 env HUGINN_PERSISTENT_TERMINAL.
    use_persistent_terminal: bool | None = Field(
        default=None,
        description=(
            "If True, dispatch via PersistentTerminal (long session, async poll). "
            "If False, force in-process dispatch. "
            "None = follow env HUGINN_PERSISTENT_TERMINAL (1=on, else off)."
        ),
    )
    # dispatch_parallel 用: [{spec_name, task}, ...], 最多 4 个 (硬 cap)
    tasks: list[dict[str, str]] | None = Field(
        default=None,
        description=(
            "For dispatch_parallel: list of {spec_name, task} dicts (1-4 items). "
            "Tasks run concurrently via asyncio.gather."
        ),
    )
    # dispatch_parallel 用: [(u_name, v_name), ...] 任务依赖. u_name/v_name 引用
    # tasks 里 spec_name+task 的标识 (ponytail: 用 task 字符串前 20 字符做 ID).
    dependencies: list[tuple[str, str]] | None = Field(
        default=None,
        description=(
            "For dispatch_parallel: task dependencies as [(u, v), ...]. "
            "u must finish before v starts. Enables DAG-aware scheduling. "
            "Omit for full parallel."
        ),
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
        "analyst (analyze data/results), support (heavy lifting in isolation, "
        "returns structured JSON findings — Oxelra Core+Support pattern)."
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
        if args.action == "dispatch_parallel":
            return await self._dispatch_parallel(args, context)

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
        # v7: 透传父 agent 的 approval_callback, 子 agent 调 ASK 工具 (vasp_tool 等) 才能拿到批准.
        dispatch_ctx.setdefault("approval_callback", context.approval_callback)

        # G1: 从 contextvar 读当前递归深度, 透传给 dispatch 守卫.
        # 主 agent 这里读到 0, 子 agent 那里读到 1+.
        from huginn.agents.subagent import _current_depth
        _depth = _current_depth.get()

        # forward subagent intermediate states to the WS via progress_cb
        from huginn.types import progress_cb

        async def _on_state(state: dict) -> None:
            cb = progress_cb.get()
            if cb is None:
                return
            msgs = state.get("messages", [])
            if not msgs:
                return
            last = msgs[-1]
            # tool calls
            if hasattr(last, "tool_calls") and last.tool_calls:
                for tc in last.tool_calls:
                    await cb({
                        "type": "subagent_event",
                        "event": "tool_call",
                        "spec": args.spec_name,
                        "tool": tc.get("name", "unknown"),
                    })
            # assistant text (truncated)
            elif hasattr(last, "content") and last.content:
                text = last.content if isinstance(last.content, str) else str(last.content)
                if len(text) > 200:
                    text = text[:200] + "..."
                await cb({
                    "type": "subagent_event",
                    "event": "text",
                    "spec": args.spec_name,
                    "text": text,
                })

        result = await self._dispatch.dispatch(
            args.spec_name, args.task, dispatch_ctx,
            on_state=_on_state, _depth=_depth,
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

    async def _dispatch_parallel(
        self, args: SubagentToolInput, context: ToolContext
    ) -> ToolResult:
        """DAG-aware 并行 dispatch.

        无 dependencies: 全部 asyncio.gather 并行.
        有 dependencies: 用 TaskDAG 拓扑分层, 同层并行, 层间串行.

        ponytail: 硬 cap 4 并行 (API 限速 + 调试可行性). DAG 调度复用 TaskDAG.
        """
        import asyncio

        if not args.tasks:
            return self._missing_field("tasks")
        if len(args.tasks) > 4:
            return ToolResult(
                data=SubagentToolOutput(
                    success=False, action="dispatch_parallel",
                    error=f"tasks 最多 4 个, got {len(args.tasks)}",
                ).model_dump(),
                success=False,
                error="tasks exceeds cap of 4",
            )
        # 校验每个 task dict 有 spec_name + task
        for i, t in enumerate(args.tasks):
            if "spec_name" not in t or "task" not in t:
                return ToolResult(
                    data=SubagentToolOutput(
                        success=False, action="dispatch_parallel",
                        error=f"tasks[{i}] 缺 spec_name 或 task",
                    ).model_dump(),
                    success=False,
                    error=f"tasks[{i}] missing spec_name or task",
                )

        dispatch_ctx = dict(args.context)
        dispatch_ctx.setdefault("agent_factory", context.agent_factory)
        dispatch_ctx.setdefault("session_id", context.session_id)
        dispatch_ctx.setdefault("workspace", context.workspace)
        dispatch_ctx.setdefault("approval_callback", context.approval_callback)
        # G1: 从 contextvar 读递归深度 (跟 _dispatch_subagent 一致)
        from huginn.agents.subagent import _current_depth
        _depth = _current_depth.get()

        # 无 dependencies: 全并行
        if not args.dependencies:
            coros = [
                self._dispatch.dispatch(
                    t["spec_name"], t["task"], dispatch_ctx, _depth=_depth,
                )
                for t in args.tasks
            ]
            results = await asyncio.gather(*coros, return_exceptions=True)
            out_results = []
            for r in results:
                if isinstance(r, Exception):
                    out_results.append({"success": False, "error": str(r)})
                else:
                    out_results.append(r.to_dict())
            return ToolResult(
                data={"action": "dispatch_parallel", "results": out_results, "n": len(out_results)},
                success=True,
            )

        # 有 dependencies: DAG 分层调度 (极限模式才开)
        import os
        if os.environ.get("HUGINN_EXTREME_DISPATCH", "0").lower() not in ("1", "true"):
            return ToolResult(
                data=SubagentToolOutput(
                    success=False, action="dispatch_parallel",
                    error="DAG-aware dispatch 需开启极限模式 (HUGINN_EXTREME_DISPATCH=1)",
                ).model_dump(),
                success=False,
                error="DAG dispatch requires HUGINN_EXTREME_DISPATCH=1",
            )
        from huginn.agents.task_dag import TaskDAG
        # task ID = spec_name + task 前 20 字符 (ponytail: 不引入显式 ID 字段)
        task_ids = [f"{t['spec_name']}:{t['task'][:20]}" for t in args.tasks]
        try:
            dag = TaskDAG(tasks=task_ids, dependencies=args.dependencies)
        except ValueError as e:
            return ToolResult(
                data=SubagentToolOutput(
                    success=False, action="dispatch_parallel", error=f"DAG 错误: {e}",
                ).model_dump(),
                success=False,
                error=str(e),
            )
        layers = dag.parallel_layers()
        id_to_task = dict(zip(task_ids, args.tasks))
        all_results: list[dict] = []
        for layer in layers:
            coros = [
                self._dispatch.dispatch(
                    id_to_task[tid]["spec_name"],
                    id_to_task[tid]["task"],
                    dispatch_ctx, _depth=_depth,
                )
                for tid in layer
            ]
            layer_results = await asyncio.gather(*coros, return_exceptions=True)
            for r in layer_results:
                if isinstance(r, Exception):
                    all_results.append({"success": False, "error": str(r)})
                else:
                    all_results.append(r.to_dict())
        return ToolResult(
            data={"action": "dispatch_parallel", "results": all_results, "n": len(all_results), "layers": layers},
            success=True,
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
