"""Plan Store 工具: 暴露 PlanStore 给 agent 和 HTTP 客户端.

planner agent 调 propose 提交结构化计划, 用户/agent 调 confirm 确认,
executor 调 get 读已确认计划, advance_step 上报步骤进度.

actions:
- propose:       提交计划 (objective + steps), 返回 plan_id, 状态 draft
- confirm:       确认 draft 计划, 释放给 executor
- reject:        拒绝 draft 计划, 附原因
- status:        查看所有计划概况
- list_pending:  列出所有 draft 计划
- list_all:      列出所有计划
- get:           取单个计划详情
- advance_step:  更新某步骤的状态/结果/错误
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.autoloop.plan_store import PlanStep, PlanStore
from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult, ValidationResult

logger = logging.getLogger(__name__)

# 默认走 $HUGINN_CACHE_DIR/plans.json, 跟 GoalScheduler 同目录.
# 测试可以通过 _init_kwargs_map 注入 path, 或直接构造时传 store.
_store_instance: PlanStore | None = None


def _get_default_store() -> PlanStore:
    global _store_instance
    if _store_instance is None:
        _store_instance = PlanStore()
    return _store_instance


class PlanStoreInput(BaseModel):
    action: Literal[
        "propose",
        "confirm",
        "reject",
        "status",
        "list_pending",
        "list_all",
        "get",
        "advance_step",
    ] = Field(...)
    # propose
    objective: str | None = Field(default=None, description="任务目标")
    steps: list[dict[str, Any]] | None = Field(
        default=None, description="PlanStep dict 列表"
    )
    auto_confirm: bool = Field(default=False, description="propose 时是否自动确认")
    # confirm / reject / get / advance_step
    plan_id: str | None = Field(default=None, description="计划ID")
    reject_reason: str | None = Field(default=None, description="拒绝原因")
    # advance_step
    step_id: str | None = Field(default=None, description="步骤ID")
    step_status: str | None = Field(
        default=None, description="running | done | error | skipped"
    )
    step_result: str | None = Field(default=None, description="步骤结果")
    step_error: str | None = Field(default=None, description="步骤错误信息")


class PlanStoreTool(HuginnTool):
    """Plan Store 工具: 结构化计划的提交/确认/执行追踪."""

    name = "plan_store_tool"
    category = "planning"
    description = (
        "Submit, confirm, reject, and track structured plans with dependencies. "
        "Actions: propose (submit plan), confirm (approve draft), "
        "reject (reject with reason), status (overview), "
        "list_pending (draft plans), list_all, get (single plan), "
        "advance_step (update step status)."
    )
    input_schema = PlanStoreInput

    def __init__(self, store: PlanStore | None = None) -> None:
        self._store = store

    def _resolve_store(self) -> PlanStore:
        return self._store or _get_default_store()

    def is_read_only(self, args: PlanStoreInput) -> bool:
        # 只读 action: status / list_pending / list_all / get
        return args.action in ("status", "list_pending", "list_all", "get")

    async def validate_input(
        self, args: PlanStoreInput, context: ToolContext | None = None
    ) -> ValidationResult:
        if args.action == "propose":
            if not args.objective:
                return ValidationResult(result=False, message="propose 需要 objective")
            if not args.steps:
                return ValidationResult(result=False, message="propose 需要 steps 列表")
        if args.action in ("confirm", "reject", "get", "advance_step"):
            if not args.plan_id:
                return ValidationResult(
                    result=False, message=f"{args.action} 需要 plan_id"
                )
        if args.action == "advance_step" and not args.step_id:
            return ValidationResult(
                result=False, message="advance_step 需要 step_id"
            )
        return ValidationResult(result=True)

    async def call(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        input_data = PlanStoreInput(**args)
        store = self._resolve_store()

        try:
            if input_data.action == "propose":
                steps = [PlanStep.from_dict(s) for s in (input_data.steps or [])]
                plan = store.create_plan(
                    objective=input_data.objective or "",
                    steps=steps,
                    auto_confirm=input_data.auto_confirm,
                )
                if input_data.auto_confirm:
                    store.confirm_plan(plan.id)
                    plan = store.get_plan(plan.id)
                return ToolResult(
                    data={
                        "plan_id": plan.id,
                        "status": plan.status,
                        "steps": [s.to_dict() for s in plan.steps],
                        "message": (
                            f"计划已提交, plan_id={plan.id}, 状态={plan.status}."
                            + (
                                " 已自动确认, 可执行."
                                if plan.status == "confirmed"
                                else " 等待用户 confirm 后执行."
                            )
                        ),
                    },
                    success=True,
                )

            if input_data.action == "confirm":
                plan = store.get_plan(input_data.plan_id or "")
                if plan is None:
                    return ToolResult(
                        data=None,
                        success=False,
                        error=f"plan_id={input_data.plan_id} 不存在",
                    )
                confirmed = store.confirm_plan(plan.id)
                return ToolResult(
                    data={
                        "plan_id": confirmed.id,
                        "status": "confirmed",
                        "message": "计划已确认, executor 可以执行.",
                    },
                    success=True,
                )

            if input_data.action == "reject":
                plan = store.get_plan(input_data.plan_id or "")
                if plan is None:
                    return ToolResult(
                        data=None,
                        success=False,
                        error=f"plan_id={input_data.plan_id} 不存在",
                    )
                rejected = store.reject_plan(plan.id, input_data.reject_reason)
                return ToolResult(
                    data={
                        "plan_id": rejected.id,
                        "status": "abandoned",
                        "reject_reason": rejected.reject_reason or "",
                        "message": "计划已拒绝, 请根据原因修改后重新 propose.",
                    },
                    success=True,
                )

            if input_data.action == "status":
                all_plans = store.list_plans()
                return ToolResult(
                    data={
                        "total": len(all_plans),
                        "by_status": _count_by_status(all_plans),
                        "pending": [
                            {"plan_id": p.id, "objective": p.objective}
                            for p in all_plans
                            if p.status == "draft"
                        ],
                    },
                    success=True,
                )

            if input_data.action == "list_pending":
                drafts = store.list_plans(status="draft")
                return ToolResult(
                    data={
                        "pending": [
                            {
                                "plan_id": p.id,
                                "objective": p.objective,
                                "n_steps": len(p.steps),
                            }
                            for p in drafts
                        ]
                    },
                    success=True,
                )

            if input_data.action == "list_all":
                all_plans = store.list_plans()
                return ToolResult(
                    data={
                        "plans": [
                            {
                                "plan_id": p.id,
                                "objective": p.objective,
                                "status": p.status,
                                "n_steps": len(p.steps),
                                "created_at": p.created_at,
                            }
                            for p in all_plans
                        ]
                    },
                    success=True,
                )

            if input_data.action == "get":
                plan = store.get_plan(input_data.plan_id or "")
                if plan is None:
                    return ToolResult(
                        data=None,
                        success=False,
                        error=f"plan_id={input_data.plan_id} 不存在",
                    )
                return ToolResult(data=plan.to_dict(), success=True)

            if input_data.action == "advance_step":
                plan = store.get_plan(input_data.plan_id or "")
                if plan is None:
                    return ToolResult(
                        data=None,
                        success=False,
                        error=f"plan_id={input_data.plan_id} 不存在",
                    )
                fields: dict[str, Any] = {}
                if input_data.step_status is not None:
                    fields["status"] = input_data.step_status
                if input_data.step_result is not None:
                    fields["result"] = input_data.step_result
                if input_data.step_error is not None:
                    fields["error"] = input_data.step_error
                updated = store.update_step(
                    plan.id, input_data.step_id or "", **fields
                )
                step = next(
                    (s for s in updated.steps if s.id == input_data.step_id), None
                )
                if step is None:
                    return ToolResult(
                        data=None,
                        success=False,
                        error=f"step_id={input_data.step_id} 不在 plan {plan.id} 里",
                    )
                return ToolResult(
                    data={
                        "plan_id": updated.id,
                        "step_id": step.id,
                        "status": step.status,
                        "result": step.result,
                        "error": step.error,
                    },
                    success=True,
                )

            return ToolResult(
                data=None,
                success=False,
                error=f"未知 action: {input_data.action}",
            )

        except Exception as e:
            logger.warning("plan_store_tool failed: %s", e, exc_info=True)
            return ToolResult(data=None, success=False, error=str(e))


def _count_by_status(plans: list) -> dict[str, int]:
    counts: dict[str, int] = {}
    for p in plans:
        counts[p.status] = counts.get(p.status, 0) + 1
    return counts
