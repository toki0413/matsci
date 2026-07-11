"""多模型团队 endpoints.

两套接口并存:
  - /team/v2/*  使用新的 ModelTeam (按 ModelCaps 把不同 LLM 路由到不同角色)
  - /team/*     保留老的 Orchestrator 接口, 向后兼容

前端应优先使用 /team/v2/* .
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from huginn.config import get_config
from huginn.pet import PetMood, get_pet_bus
from huginn.server_core import get_agent_factory, get_orchestrator, get_plan_store

router = APIRouter(tags=["team"])


# ── 新接口: 多模型团队 (ModelTeam) ─────────────────────────────


def _build_model_team():
    """根据当前 config 组建 ModelTeam.

    每次请求都重新组建, 这样前端改完 config 立刻能看到新阵容.
    ModelTeam.from_config 只读配置, 不会真起 LLM, 代价很小.
    """
    from huginn.agents.team import ModelTeam

    cfg = get_config()
    return ModelTeam.from_config(cfg)


@router.get("/team/v2/members")
async def team_v2_members() -> dict[str, Any]:
    """列出多模型团队的成员 (角色 / 模型 / 能力).

    和老的 /team/profiles 不同: 这里按 TeamRole 展示,
    一个角色可能由不同 profile 承担, 也能看出单模型多角色的退化情况.
    """
    try:
        team = _build_model_team()
        members = team.list_members()
        # 标注是否单模型兼容模式 (所有成员 profile 相同)
        profiles = {m.get("profile") for m in members}
        return {
            "success": True,
            "single_model_mode": len(profiles) <= 1,
            "members": members,
            "roles_covered": sorted({m["role"] for m in members}),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/team/v2/plan")
async def team_v2_plan(params: dict[str, Any]) -> dict[str, Any]:
    """只用 planner 成员生成执行计划, 不真正执行.

    返回 planner 输出的原始文本 + 解析后的步骤列表.
    前端可以拿这个让用户确认再走 /team/v2/run.
    """
    objective = params.get("objective", "")
    if not objective:
        return {"success": False, "error": "objective is required"}

    try:
        team = _build_model_team()
        from huginn.agents.team import TeamRole

        if TeamRole.PLANNER not in team.members:
            return {
                "success": False,
                "error": "当前团队没有 planner 角色, 请先配置一个带 reasoning 能力的模型",
            }

        # 只跑 planner 一步, 不进入 _execute_plan
        traces: list = []
        ctx = {"original_task": objective}
        plan_text = await team._delegate(TeamRole.PLANNER, objective, ctx, traces)
        steps = team._parse_plan(plan_text)
        if not steps:
            steps = team._default_plan(objective)

        return {
            "success": True,
            "objective": objective,
            "planner_output": plan_text,
            "plan_text": team._plan_to_text(steps),
            "steps": [
                {
                    "id": s.id,
                    "role": s.role.value,
                    "task": s.task,
                    "depends_on": list(s.depends_on),
                }
                for s in steps
            ],
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/team/v2/run")
async def team_v2_run(params: dict[str, Any]) -> dict[str, Any]:
    """跑完整流水线: 规划 → 按步骤执行 → 审查.

    会真的调用每个成员的 LLM, 耗时取决于步骤数和模型响应速度.
    进度通过 pet bus 发布事件, 前端可订阅 SSE 看到 mood 变化.
    """
    objective = params.get("objective", "")
    if not objective:
        return {"success": False, "error": "objective is required"}

    # 可选: 让前端指定 thread_id 以保留对话上下文
    thread_id = params.get("thread_id")

    bus = get_pet_bus()

    def _publish(mood: PetMood, msg: str, details: dict | None = None) -> None:
        try:
            bus.publish(mood=mood, message=msg, details=details or {})
        except Exception:
            # pet bus 不应该影响主流程
            pass

    try:
        team = _build_model_team()
        _publish(PetMood.WORKING, f"团队启动: {len(team.members)} 个成员")

        ctx: dict[str, Any] = {}
        if thread_id:
            ctx["thread_id"] = thread_id

        result = await team.run(objective, context=ctx)
        _publish(
            PetMood.SUCCESS if result.get("final_output") else PetMood.ERROR,
            f"团队完成 · {len(result.get('trace', []))} 步",
        )
        return {"success": True, **result}
    except Exception as e:
        _publish(PetMood.ERROR, f"团队执行失败: {e}")
        return {"success": False, "error": str(e)}


# ── 持久化计划接口: PlanStore lifecycle ─────────────────────────


@router.post("/team/v2/plans")
async def create_plan(params: dict[str, Any]) -> dict[str, Any]:
    """用 planner 角色分解目标, 持久化到 PlanStore.

    auto_confirm=True 时直接进入 confirmed 状态可执行;
    默认 False 返回 draft, 需要用户走 /confirm 确认后再执行.
    """
    objective = params.get("objective", "")
    if not objective:
        return {"success": False, "error": "objective is required"}
    auto_confirm = bool(params.get("auto_confirm", False))
    try:
        orch = get_orchestrator()
        if orch.plan_store is None:
            return {"success": False, "error": "plan_store not configured"}
        plan = await orch.plan(objective, auto_confirm=auto_confirm)
        return {"success": True, "plan": plan.to_dict()}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/team/v2/plans")
async def list_plans(status: str | None = None) -> dict[str, Any]:
    """列出所有计划, 可选按 status 过滤 (draft/confirmed/executing/completed/failed/abandoned)."""
    try:
        store = get_plan_store()
        plans = store.list_plans(status=status)
        return {"success": True, "plans": [p.to_dict() for p in plans]}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/team/v2/plans/{plan_id}")
async def get_plan(plan_id: str) -> dict[str, Any]:
    """按 id 取单个计划."""
    try:
        store = get_plan_store()
        plan = store.get_plan(plan_id)
        if plan is None:
            return {"success": False, "error": "plan not found"}
        return {"success": True, "plan": plan.to_dict()}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/team/v2/plans/{plan_id}/confirm")
async def confirm_plan(plan_id: str) -> dict[str, Any]:
    """把 draft 计划确认成 confirmed, 之后才能执行."""
    try:
        store = get_plan_store()
        plan = store.get_plan(plan_id)
        if plan is None:
            return {"success": False, "error": "plan not found"}
        plan = store.confirm_plan(plan_id)
        return {"success": True, "plan": plan.to_dict()}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/team/v2/plans/{plan_id}/reject")
async def reject_plan(plan_id: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """拒绝 draft 计划, 标记为 abandoned. 可带 reason."""
    try:
        store = get_plan_store()
        plan = store.get_plan(plan_id)
        if plan is None:
            return {"success": False, "error": "plan not found"}
        reason = (params or {}).get("reason")
        plan = store.reject_plan(plan_id, reason=reason)
        return {"success": True, "plan": plan.to_dict()}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/team/v2/plans/{plan_id}/execute")
async def execute_plan(plan_id: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """执行已确认的计划. 走 Orchestrator.execute(plan_id).

    执行完成后返回最终 plan 状态 + result 摘要.
    """
    try:
        store = get_plan_store()
        orch = get_orchestrator()
        if orch.plan_store is None:
            return {"success": False, "error": "plan_store not configured"}

        result = await orch.execute(plan_id)
        plan = store.get_plan(plan_id)
        return {
            "success": True,
            "plan": plan.to_dict() if plan else None,
            "result": {
                "summary": result.summary,
                "outputs": result.outputs,
                "success": result.success,
                "error": result.error,
            },
        }
    except Exception as e:
        # 执行可能抛错 (plan 不存在 / 状态不对), 顺带返回当前 plan 状态
        try:
            plan = get_plan_store().get_plan(plan_id)
            return {"success": False, "error": str(e), "plan": plan.to_dict() if plan else None}
        except Exception:
            return {"success": False, "error": str(e)}


@router.delete("/team/v2/plans/{plan_id}")
async def delete_plan(plan_id: str) -> dict[str, Any]:
    """删除计划."""
    try:
        store = get_plan_store()
        deleted = store.delete_plan(plan_id)
        if not deleted:
            return {"success": False, "error": "plan not found"}
        return {"success": True, "deleted": plan_id}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── 老接口: Orchestrator (保留向后兼容) ─────────────────────────


@router.post("/team/v2/fusion")
async def team_v2_fusion(params: dict[str, Any]) -> dict[str, Any]:
    """Fusion 模式: 并行多模型回答 → 裁判合成.

    和 /team/v2/run 的区别:
    - run: 串行流水线 (plan → execute → review)
    - fusion: 并行 fan-out → 合成 fan-in

    适合开放性研究问题, 不适合有明确步骤的执行任务.

    参数:
        query: 要查询的问题
        thread_id: 可选, 用于对话上下文
        panel_roles: 可选, 指定参与并行回答的角色列表
        synthesizer_role: 可选, 负责合成的角色 (默认 critic)
        max_panel: 可选, 最多并行成员数 (默认 5)
    """
    query = params.get("query") or params.get("objective") or ""
    if not query:
        return {"success": False, "error": "query is required"}

    thread_id = params.get("thread_id")
    panel_roles_raw = params.get("panel_roles")
    synthesizer_role_raw = params.get("synthesizer_role", "critic")
    max_panel = int(params.get("max_panel", 5))
    rounds = max(1, int(params.get("rounds", 1)))

    bus = get_pet_bus()

    def _publish(mood: PetMood, msg: str, details: dict | None = None) -> None:
        try:
            bus.publish(mood=mood, message=msg, details=details or {})
        except Exception:
            pass

    try:
        from huginn.agents.team import TeamRole

        # 解析角色参数
        panel_roles = None
        if panel_roles_raw and isinstance(panel_roles_raw, list):
            try:
                panel_roles = [TeamRole(r) for r in panel_roles_raw]
            except ValueError:
                pass

        try:
            synthesizer_role = TeamRole(synthesizer_role_raw)
        except ValueError:
            synthesizer_role = TeamRole.CRITIC

        team = _build_model_team()
        _publish(
            PetMood.WORKING,
            f"Fusion 启动: {len(team.members)} 个成员, panel ≤ {max_panel}, {rounds} 轮",
        )

        ctx: dict[str, Any] = {}
        if thread_id:
            ctx["thread_id"] = thread_id

        result = await team.fusion_query(
            query,
            ctx,
            panel_roles=panel_roles,
            synthesizer_role=synthesizer_role,
            max_panel=max_panel,
            rounds=rounds,
        )

        panel_count = len(result.get("panel_responses", []))
        _publish(
            PetMood.SUCCESS if result.get("final_answer") else PetMood.ERROR,
            f"Fusion 完成 · {panel_count} 个 panel 回答已合成",
        )

        return {"success": True, **result}
    except Exception as e:
        _publish(PetMood.ERROR, f"Fusion 执行失败: {e}")
        return {"success": False, "error": str(e)}


# ── 老接口: Orchestrator (保留向后兼容) ─────────────────────────


@router.get("/team/profiles")
async def team_profiles() -> dict[str, Any]:
    """List enabled agent profiles available for team tasks."""
    try:
        factory = get_agent_factory()
        profiles = [
            {
                "id": p.id,
                "name": p.name or p.id,
                "model_alias": p.model_alias,
                "persona": p.persona,
                "tools": p.tools,
                "enabled": p.enabled,
            }
            for p in factory.list_profiles()
        ]
        return {"profiles": profiles}
    except Exception as e:
        return {"error": str(e)}


@router.post("/team/plan")
async def team_plan(params: dict[str, Any]) -> dict[str, Any]:
    """Ask the lead agent to break an objective into subtasks."""
    objective = params.get("objective", "")
    if not objective:
        return {"success": False, "error": "objective is required"}
    try:
        orchestrator = get_orchestrator()
        plan = await orchestrator.plan(objective)
        return {
            "success": True,
            "objective": plan.objective,
            "tasks": [
                {
                    "task_id": t.task_id,
                    "agent_id": t.agent_id,
                    "prompt": t.prompt,
                    "depends_on": t.depends_on,
                }
                for t in plan.tasks
            ],
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/team/run")
async def team_run(params: dict[str, Any]) -> dict[str, Any]:
    """Run a multi-agent plan and return the synthesized result."""
    from huginn.agents.orchestrator import SubTask

    objective = params.get("objective", "")
    if not objective:
        return {"success": False, "error": "objective is required"}

    async def on_status(task: SubTask) -> None:
        mood = (
            PetMood.WORKING
            if task.status == "running"
            else PetMood.SUCCESS if task.status == "done" else PetMood.ERROR
        )
        get_pet_bus().publish(
            mood=mood,
            message=f"{task.task_id} ({task.agent_id}): {task.status}",
            details={
                "task_id": task.task_id,
                "agent_id": task.agent_id,
                "status": task.status,
            },
        )

    try:
        orchestrator = get_orchestrator()
        result = await orchestrator.run(objective, on_status=on_status)
        return {
            "success": result.success,
            "objective": result.objective,
            "summary": result.summary,
            "outputs": result.outputs,
            "error": result.error,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
