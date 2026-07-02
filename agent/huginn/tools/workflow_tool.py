"""workflow_tool — 让 agent 提交/查询/取消/收集动态并行工作流.

A5 (W3): agent 写一个声明式脚本 (N 个独立 subtask, 每个是一次 tool call),
orchestrator 并发跑 (默认 8 并发), 失败的 subtask 不炸整个 workflow.
参考 Claude Code 的 Dynamic Workflows, 但用 Python/asyncio 而非 JS.

actions:
- submit_script: 解析脚本 JSON, 后台启动, 返回 workflow_id + 初始状态
- status:       查运行中 workflow 的进度摘要
- cancel:       取消运行中 workflow (已完成的返回 False)
- collect:      拿聚合结果 (可选阻塞等待完成)
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.autoloop.dynamic_workflow import (
    WorkflowScript,
    get_shared_workflow_registry,
)
from huginn.phases import ResearchPhase
from huginn.tools.base import HuginnTool
from huginn.tools.profile import ToolProfile
from huginn.types import ToolContext, ToolResult


class WorkflowToolInput(BaseModel):
    action: Literal["submit_script", "status", "cancel", "collect"] = Field(
        description=(
            "submit_script: 提交并行工作流脚本, 后台启动; "
            "status: 查进度摘要; "
            "cancel: 取消运行中工作流; "
            "collect: 拿聚合结果 (可阻塞等待)"
        )
    )
    script: dict[str, Any] | None = Field(
        default=None,
        description=(
            "submit_script 必填. 工作流脚本: "
            '{"objective": "...", "max_concurrent": 8, "subtasks": ['
            '{"id": "s1", "tool": "vasp_tool", "args": {...}}, ...]}'
        ),
    )
    workflow_id: str | None = Field(
        default=None,
        description="status / cancel / collect 必填: 要操作的工作流 id.",
    )
    timeout: float | None = Field(
        default=None,
        description="collect 可选: 阻塞等待完成的超时秒数. None=不阻塞 (立即返回当前状态).",
    )


class WorkflowTool(HuginnTool):
    """动态并行工作流的提交/查询/取消/收集."""

    name = "workflow_tool"
    category = "meta"
    description = (
        "提交并行工作流脚本 (多个独立 tool call 并发执行), 查询进度, "
        "取消, 或收集聚合结果. 失败的 subtask 不影响其他 subtask."
    )
    input_schema = WorkflowToolInput
    profile = ToolProfile(
        cost_tier="light",
        phases=frozenset({ResearchPhase.EXECUTION, ResearchPhase.OPEN}),
    )

    async def call(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        # Pydantic schema 校验在 try 外, ValidationError 传播不吞
        input_data = WorkflowToolInput(**args)
        try:
            if input_data.action == "submit_script":
                return self._submit_script(input_data, context)
            if input_data.action == "status":
                return self._status(input_data)
            if input_data.action == "cancel":
                return self._cancel(input_data)
            if input_data.action == "collect":
                return await self._collect(input_data)
            return ToolResult(
                data=None, success=False,
                error=f"未知 action: {input_data.action}",
            )
        except Exception as exc:
            return ToolResult(
                data=None, success=False,
                error=f"Workflow tool failed: {exc}",
            )

    # ── action 实现 ──────────────────────────────────────────────

    def _submit_script(
        self, input_data: WorkflowToolInput, context: ToolContext | None
    ) -> ToolResult:
        if not input_data.script:
            return ToolResult(
                data=None, success=False,
                error="submit_script 需要 script 参数",
            )
        try:
            script = WorkflowScript.from_dict(input_data.script)
        except (ValueError, TypeError) as exc:
            return ToolResult(
                data=None, success=False,
                error=f"脚本解析失败: {exc}",
            )
        if not script.subtasks:
            return ToolResult(
                data=None, success=False,
                error="脚本无有效 subtask (每个 subtask 需有 tool 名)",
            )
        registry = get_shared_workflow_registry()
        result = registry.submit(script, context)
        return ToolResult(
            data={
                "workflow_id": script.id,
                "status": result.status,
                "n_subtasks": len(script.subtasks),
                "objective": script.objective,
                "max_concurrent": script.max_concurrent,
            },
            success=True,
        )

    def _status(self, input_data: WorkflowToolInput) -> ToolResult:
        if not input_data.workflow_id:
            return ToolResult(
                data=None, success=False,
                error="status 需要 workflow_id",
            )
        registry = get_shared_workflow_registry()
        result = registry.get(input_data.workflow_id)
        if result is None:
            return ToolResult(
                data=None, success=False,
                error=f"workflow '{input_data.workflow_id}' 不存在",
            )
        return ToolResult(data=result.summary(), success=True)

    def _cancel(self, input_data: WorkflowToolInput) -> ToolResult:
        if not input_data.workflow_id:
            return ToolResult(
                data=None, success=False,
                error="cancel 需要 workflow_id",
            )
        registry = get_shared_workflow_registry()
        ok = registry.cancel(input_data.workflow_id)
        if not ok:
            result = registry.get(input_data.workflow_id)
            status = result.status if result else "not_found"
            return ToolResult(
                data={"workflow_id": input_data.workflow_id, "cancelled": False, "status": status},
                success=False,
                error=f"无法取消 (状态: {status})",
            )
        return ToolResult(
            data={"workflow_id": input_data.workflow_id, "cancelled": True},
            success=True,
        )

    async def _collect(self, input_data: WorkflowToolInput) -> ToolResult:
        if not input_data.workflow_id:
            return ToolResult(
                data=None, success=False,
                error="collect 需要 workflow_id",
            )
        registry = get_shared_workflow_registry()
        # collect() 统一处理三种情况: 已完成直接返回, 后台任务在跑就等,
        # 没后台任务 (或任务已死) 就同步跑一遍. timeout 只对"等后台任务"生效.
        result = await registry.collect(
            input_data.workflow_id, timeout=input_data.timeout
        )
        if result is None:
            return ToolResult(
                data=None, success=False,
                error=f"workflow '{input_data.workflow_id}' 不存在",
            )
        return ToolResult(data=result.to_dict(), success=True)
