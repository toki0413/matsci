"""Plan-build mode endpoint."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter
from langchain_core.messages import ToolMessage

from huginn.server_core import get_context, get_planner_agent

router = APIRouter(tags=["planner"])

logger = logging.getLogger(__name__)


@router.post("/plan")
async def generate_plan(params: dict[str, Any]) -> dict[str, Any]:
    """Generate a step-by-step plan without executing any tools."""
    agent = get_planner_agent()
    if agent.model is None:
        return {
            "error": "No LLM configured. Set provider and API key to generate plans."
        }

    content = params.get("content", "")
    thread_id = params.get("thread_id", "plan")
    if not content.strip():
        return {"error": "content is required"}

    # Optionally ground the plan with codebase search results
    if get_context().codebase is not None:
        try:
            results = await asyncio.to_thread(
                get_context().codebase.search, content, top_k=3
            )
            if results:
                ctx = "\n\n".join(
                    f"[{i+1}] {r['path']}\n{r['text']}" for i, r in enumerate(results)
                )
                content = (
                    "Use the following relevant codebase snippets to inform your plan. "
                    "Do not execute any actions; just plan.\n\n"
                    f"{ctx}\n\n"
                    f"Request: {content}"
                )
        except Exception:
            logger.warning("planner codebase search failed", exc_info=True)

    try:
        full_response = ""
        async for state in agent.chat(content, thread_id):
            msgs = state.get("messages", [])
            if msgs:
                last = msgs[-1]
                if hasattr(last, "content") and not isinstance(last, ToolMessage):
                    full_response = last.content
        return {"plan": full_response}
    except Exception as e:
        logger.error("unexpected error", exc_info=True)
        return {"error": f"Planner error: {str(e)}"}


@router.post("/plan/propose")
async def propose_plan(params: dict[str, Any]) -> dict[str, Any]:
    """提交一份设计计划, 进入 pending 审批态.

    复用 design_plan_tool 的 _PlanStore, 让前端/外部等审批人能把计划提进去,
    后续 confirm/reject 走同一存储, 门闸 (DesignPlanGateHook) 才能识别.
    thread_id 用于隔离确认状态, 必须跟 agent chat 用的 thread_id 一致.
    """
    from huginn.tools.design.design_plan_tool import _PlanStore

    goal = params.get("goal", "")
    if not goal or not goal.strip():
        return {"error": "goal is required"}

    plan = {
        "goal": goal.strip(),
        "layout": params.get("layout", ""),
        "style": params.get("style", ""),
        "steps": params.get("steps", []) or [],
        "parameters": params.get("parameters", {}) or {},
        "tools": params.get("tools", []) or [],
        "expected_output": params.get("expected_output", ""),
    }
    plan_id = _PlanStore.instance().propose(plan)
    return {
        "plan_id": plan_id,
        "status": "pending",
        "plan": plan,
        "thread_id": params.get("thread_id"),
    }


@router.get("/plans/pending")
async def list_pending_plans() -> dict[str, Any]:
    """列出所有 pending 审批态计划."""
    from huginn.tools.design.design_plan_tool import _PlanStore

    return {"pending": _PlanStore.instance().list_pending()}


@router.get("/plan/status")
async def plan_status(thread_id: str | None = None) -> dict[str, Any]:
    """查看某 thread 的计划审批态 (是否有 confirmed plan, pending 列表等)."""
    from huginn.tools.design.design_plan_tool import _PlanStore

    store = _PlanStore.instance()
    return {
        "thread_id": thread_id,
        "has_confirmed": store.has_confirmed(thread_id),
        "last_confirmed": store.last_confirmed(thread_id),
        "pending": store.list_pending(),
        "total_plans": len(store._plans),
    }


@router.post("/plan/{plan_id}/confirm")
async def confirm_plan(plan_id: str, params: dict[str, Any]) -> dict[str, Any]:
    """审批通过一条 pending 计划, 释放对应 thread 的门闸."""
    from huginn.tools.design.design_plan_tool import _PlanStore

    thread_id = params.get("thread_id")
    ok = _PlanStore.instance().confirm(plan_id, thread_id)
    if not ok:
        return {"success": False, "error": f"plan_id={plan_id} 不存在"}
    return {
        "success": True,
        "plan_id": plan_id,
        "status": "confirmed",
        "thread_id": thread_id,
    }


@router.post("/plan/{plan_id}/reject")
async def reject_plan(plan_id: str, params: dict[str, Any]) -> dict[str, Any]:
    """拒绝一条计划, 回收对应 thread 的门闸, 重新走 propose."""
    from huginn.tools.design.design_plan_tool import _PlanStore

    thread_id = params.get("thread_id")
    reason = params.get("reject_reason")
    ok = _PlanStore.instance().reject(plan_id, reason, thread_id)
    if not ok:
        return {"success": False, "error": f"plan_id={plan_id} 不存在"}
    return {
        "success": True,
        "plan_id": plan_id,
        "status": "rejected",
        "reject_reason": reason or "",
        "thread_id": thread_id,
    }
