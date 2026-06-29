"""Design Plan 工具: 先计划, 再执行.

设计思路 (参考 v0/nudge/DesignPlan 机制):
- agent 在生成前输出结构化设计计划 (布局/风格/内容层级/步骤/参数)
- 用户确认后才允许执行, 方向对齐在前
- 计划与确认状态存在内存里, 跨工具调用共享
- 配合 DesignPlanGateHook (PRE_TOOL_USE) 拦截执行类工具

actions:
- propose:  agent 提交计划, 返回 plan_id, 状态置 pending
- confirm:  用户确认计划, 状态置 confirmed, 释放 gate
- reject:   用户拒绝, 状态置 rejected, 要求 agent 重新 propose
- status:   查看当前计划与状态
- list_pending: 列出所有 pending 计划

计划结构:
{
    "goal": "用户意图",
    "layout": "布局描述 (网格/层级/区块划分)",
    "style": "风格描述 (现代/简约/学术/工业)",
    "steps": ["步骤1", "步骤2", ...],
    "parameters": {"key": "value", ...},
    "tools": ["vasp_tool", "lammps_tool", ...],
    "expected_output": "预期产出描述",
}
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult, ValidationResult

logger = logging.getLogger(__name__)

# 被gate拦截的执行类工具. 这些工具会消耗计算资源或产生副作用,
# 必须先有用户确认的 plan 才能调用.
GATED_TOOLS: frozenset[str] = frozenset({
    "vasp_tool",
    "lammps_tool",
    "qe_tool",
    "cp2k_tool",
    "comsol_tool",
    "abaqus_tool",
    "openfoam_tool",
    "orchestrate",
    "job_tool",
    "high_throughput_tool",
    "ml_potential_tool",
    "packing_tool",
})


class DesignPlanInput(BaseModel):
    action: Literal["propose", "confirm", "reject", "status", "list_pending"] = (
        Field(...)
    )
    # propose 时必填
    goal: str | None = Field(default=None, description="用户意图/任务目标")
    layout: str | None = Field(default=None, description="布局描述")
    style: str | None = Field(default=None, description="风格描述")
    steps: list[str] | None = Field(default=None, description="执行步骤列表")
    parameters: dict[str, Any] | None = Field(
        default=None, description="关键参数"
    )
    tools: list[str] | None = Field(
        default=None, description="计划使用的工具列表"
    )
    expected_output: str | None = Field(default=None, description="预期产出")
    # confirm/reject 时必填
    plan_id: str | None = Field(default=None, description="计划ID")
    reject_reason: str | None = Field(
        default=None, description="拒绝原因 (reject 时填)"
    )
    # 会话/线程标识, confirm 状态按它隔离. HTTP 直调 confirm 时必须显式传,
    # 且要跟 agent chat 用的 thread_id 一致, 否则 confirm 标记到 "http"
    # 这个 thread, agent chat 时 gate 用真实 thread_id 查不到, 还是会被拦.
    thread_id: str | None = Field(
        default=None, description="会话标识, 按 thread 隔离确认状态"
    )


class _PlanStore:
    """进程内计划存储. 单例, 跨工具调用共享.

    并发安全: 所有读写都走 self._lock. 早期版本无锁, 多 thread 并发
    propose/confirm 会丢更新, list_pending 在迭代中改 dict 会炸
    'dictionary changed size during iteration'. 现在统一加锁串行化.
    """

    _instance: _PlanStore | None = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        # plan_id -> plan dict
        self._plans: dict[str, dict[str, Any]] = {}
        # 按 thread 隔离的 confirmed 状态: thread_id -> plan_id
        # 以前是单个 _last_confirmed 全局共享, 一个 thread confirm 后所有
        # thread 的 gate 都放行, 安全机制被绕过. 现在按 thread 分开存.
        # thread_id 为 None 时落到 "_global_" key, 保留旧的全局共享行为.
        self._confirmed_by_thread: dict[str, str] = {}
        self._lock = threading.Lock()

    @classmethod
    def instance(cls) -> _PlanStore:
        # 双重检查锁定: 首次并发调用 instance() 不会创建两个实例
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def propose(self, plan: dict[str, Any]) -> str:
        plan_id = f"plan-{uuid.uuid4().hex[:8]}"
        plan["plan_id"] = plan_id
        plan["status"] = "pending"
        plan["created_at"] = time.time()
        with self._lock:
            self._plans[plan_id] = plan
        return plan_id

    def confirm(self, plan_id: str, thread_id: str | None = None) -> bool:
        with self._lock:
            plan = self._plans.get(plan_id)
            if not plan:
                return False
            plan["status"] = "confirmed"
            plan["confirmed_at"] = time.time()
            # 只放行对应 thread 的 gate, 别污染别的 thread
            self._confirmed_by_thread[thread_id or "_global_"] = plan_id
        return True

    def reject(self, plan_id: str, reason: str | None, thread_id: str | None = None) -> bool:
        with self._lock:
            plan = self._plans.get(plan_id)
            if not plan:
                return False
            plan["status"] = "rejected"
            plan["reject_reason"] = reason or ""
            plan["rejected_at"] = time.time()
            # 拒绝后清掉对应 thread 的 confirmed, 该 thread 的 gate 重新生效
            key = thread_id or "_global_"
            if self._confirmed_by_thread.get(key) == plan_id:
                del self._confirmed_by_thread[key]
        return True

    def get(self, plan_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._plans.get(plan_id)

    def list_pending(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {"plan_id": pid, "goal": p.get("goal", ""), "status": p["status"]}
                for pid, p in self._plans.items()
                if p["status"] == "pending"
            ]

    def last_confirmed(self, thread_id: str | None = None) -> str | None:
        with self._lock:
            return self._confirmed_by_thread.get(thread_id or "_global_")

    def has_confirmed(self, thread_id: str | None = None) -> bool:
        with self._lock:
            return self._confirmed_by_thread.get(thread_id or "_global_") is not None


class DesignPlanTool(HuginnTool):
    """Design Plan 工具: 先计划再执行, 用户确认 gate."""

    name = "design_plan_tool"
    category = "design"
    description = (
        "Submit a structured design plan before executing heavy tasks. "
        "Actions: propose (agent submits plan), confirm (user approves), "
        "reject (user rejects with reason), status (check plan), "
        "list_pending (show pending plans). Heavy tools (VASP/LAMMPS/etc) "
        "are gated until a plan is confirmed."
    )
    input_schema = DesignPlanInput

    def is_read_only(self, args: DesignPlanInput) -> bool:
        # 所有 action 都是状态修改, 但不产生副作用, 视为 read-only 避免权限拦
        return True

    async def validate_input(
        self, args: DesignPlanInput, context: ToolContext | None = None
    ) -> ValidationResult:
        if args.action == "propose" and not args.goal:
            return ValidationResult(
                result=False, message="propose 需要 goal 字段"
            )
        if args.action in ("confirm", "reject") and not args.plan_id:
            return ValidationResult(
                result=False,
                message=f"{args.action} 需要 plan_id 字段",
            )
        return ValidationResult(result=True)

    async def call(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        input_data = DesignPlanInput(**args)
        store = _PlanStore.instance()

        # thread_id 解析: 优先用 args 里显式传的 (HTTP 直调 confirm 必须
        # 传, 跟 agent chat 的真实 thread_id 对齐); 没传就退回 context
        # 的 session_id; 都没有留 None, store 内部落到 "_global_" (旧行为).
        # ToolContext 没有 thread_id/metadata 字段, 这里只能拿 session_id.
        tid = input_data.thread_id
        if not tid and context is not None:
            tid = context.session_id or None

        try:
            if input_data.action == "propose":
                plan = {
                    "goal": input_data.goal or "",
                    "layout": input_data.layout or "",
                    "style": input_data.style or "",
                    "steps": input_data.steps or [],
                    "parameters": input_data.parameters or {},
                    "tools": input_data.tools or [],
                    "expected_output": input_data.expected_output or "",
                }
                plan_id = store.propose(plan)
                return ToolResult(
                    data={
                        "plan_id": plan_id,
                        "status": "pending",
                        "plan": plan,
                        "message": (
                            "计划已提交, 等待用户确认. "
                            "请用户调用 design_plan_tool action=confirm "
                            f"plan_id={plan_id} 确认, 或 action=reject 拒绝."
                        ),
                    },
                    success=True,
                )

            if input_data.action == "confirm":
                ok = store.confirm(input_data.plan_id or "", tid)
                if not ok:
                    return ToolResult(
                        data=None,
                        success=False,
                        error=f"plan_id={input_data.plan_id} 不存在",
                    )
                return ToolResult(
                    data={
                        "plan_id": input_data.plan_id,
                        "status": "confirmed",
                        "message": "计划已确认, 现在可以调用执行类工具.",
                    },
                    success=True,
                )

            if input_data.action == "reject":
                ok = store.reject(
                    input_data.plan_id or "", input_data.reject_reason, tid
                )
                if not ok:
                    return ToolResult(
                        data=None,
                        success=False,
                        error=f"plan_id={input_data.plan_id} 不存在",
                    )
                return ToolResult(
                    data={
                        "plan_id": input_data.plan_id,
                        "status": "rejected",
                        "reject_reason": input_data.reject_reason or "",
                        "message": (
                            "计划已被拒绝. 请根据 reject_reason 修改后 "
                            "重新 propose."
                        ),
                    },
                    success=True,
                )

            if input_data.action == "status":
                if input_data.plan_id:
                    plan = store.get(input_data.plan_id)
                    if not plan:
                        return ToolResult(
                            data=None,
                            success=False,
                            error=f"plan_id={input_data.plan_id} 不存在",
                        )
                    return ToolResult(data=plan, success=True)
                # 无 plan_id 返回总体状态
                return ToolResult(
                    data={
                        "total_plans": len(store._plans),
                        "last_confirmed": store.last_confirmed(tid),
                        "has_confirmed": store.has_confirmed(tid),
                        "pending": store.list_pending(),
                    },
                    success=True,
                )

            if input_data.action == "list_pending":
                return ToolResult(
                    data={"pending": store.list_pending()}, success=True
                )

            return ToolResult(
                data=None,
                success=False,
                error=f"未知 action: {input_data.action}",
            )

        except Exception as e:
            logger.warning("design_plan_tool failed: %s", e, exc_info=True)
            return ToolResult(data=None, success=False, error=str(e))
