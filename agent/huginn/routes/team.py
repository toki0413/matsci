"""Multi-agent team endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from huginn.pet import PetMood, get_pet_bus
from huginn.server_core import get_agent_factory, get_orchestrator

router = APIRouter(tags=["team"])


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
