"""Workflow template listing and execution endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from huginn.server_core import get_agent_factory, get_memory_manager
from huginn.tools.registry import ToolRegistry
from huginn.types import ToolContext
from huginn.workflows.engine import WorkflowEngine
from huginn.workflows.templates import get_template

router = APIRouter(tags=["workflows"])


@router.get("/workflows")
async def list_workflows() -> list[str]:
    from huginn.workflows.templates import list_templates

    return list_templates()


@router.post("/workflows/execute")
async def execute_workflow(params: dict[str, Any]) -> dict[str, Any]:
    """Execute a workflow template.

    Args:
        template: Template name (e.g., "standard_dft", "aimd")
        args: Arguments passed to the template function
    """
    template_name = params.get("template")
    template_args = params.get("args", {})

    template_fn = get_template(template_name)
    if not template_fn:
        return {"error": f"Template '{template_name}' not found"}

    try:
        stages = template_fn(**template_args)
    except Exception as e:
        return {"error": f"Failed to build workflow: {e}"}

    engine = WorkflowEngine(ToolRegistry)
    context = ToolContext(
        session_id="http",
        workspace=".",
        memory_manager=get_memory_manager(),
        agent_factory=get_agent_factory(),
    )

    result = await engine.execute(stages, context)

    return {
        "success": result.success,
        "total_walltime": result.total_walltime,
        "stages": {
            sid: {
                "name": s.name,
                "status": s.status,
                "attempts": s.attempts,
                "error": s.result.error if s.result else None,
            }
            for sid, s in result.stages.items()
        },
        "outputs": result.outputs,
        "error": result.error,
    }
