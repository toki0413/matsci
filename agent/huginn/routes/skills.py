"""Skill listing and execution endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from huginn.server_core import get_agent_factory, get_memory_manager
from huginn.skills.base import DeclarativeSkillExecutor
from huginn.skills.registry import SkillRegistry
from huginn.tools.registry import ToolRegistry
from huginn.types import ToolContext

router = APIRouter(tags=["skills"])


@router.get("/skills")
async def list_skills() -> list[dict[str, Any]]:
    """List all registered skills."""
    # Ensure presets are loaded and registered
    from huginn.skills import presets  # noqa: F401

    return [
        {
            "name": skill.name,
            "description": skill.description,
            "category": skill.category,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "description": p.description,
                    "required": p.required,
                    "default": p.default,
                }
                for p in skill.parameters
            ],
            "tags": skill.tags,
        }
        for skill in SkillRegistry.get_all_definitions()
    ]


@router.post("/skills/execute")
async def execute_skill(params: dict[str, Any]) -> dict[str, Any]:
    """Execute a skill by name with the provided arguments."""
    from huginn.skills import presets  # noqa: F401

    skill_name = params.get("skill")
    skill_args = params.get("args", {})

    skill = SkillRegistry.get(skill_name)
    if not skill:
        return {"error": f"Skill '{skill_name}' not found"}

    executor = DeclarativeSkillExecutor(ToolRegistry)
    context = ToolContext(
        session_id="http",
        workspace=".",
        memory_manager=get_memory_manager(),
        agent_factory=get_agent_factory(),
    )
    result = await executor.execute(skill, skill_args, context.__dict__)
    # Keep the response JSON-serializable: drop non-primitive objects from the
    # skill's final context (e.g. MemoryManager, AgentFactory).
    if isinstance(result, dict) and "context" in result:
        safe_types = (str, int, float, bool, type(None), list, dict, tuple)
        result["context"] = {
            k: v
            for k, v in result["context"].items()
            if isinstance(v, safe_types)
        }
    return result
