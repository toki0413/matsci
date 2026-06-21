"""Autonomous coder mode endpoint."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter

from huginn.pet import PetMood, get_pet_bus

router = APIRouter(tags=["coder"])


@router.post("/coder")
async def run_coder(params: dict[str, Any]) -> dict[str, Any]:
    """Run an autonomous coding session and return the final answer."""
    task = params.get("task", "")
    if not task:
        return {"success": False, "error": "task is required"}

    auto_approve = bool(params.get("auto_approve", False))
    max_iterations = params.get("max_iterations")
    if max_iterations is not None:
        try:
            max_iterations = int(max_iterations)
        except (TypeError, ValueError):
            return {"success": False, "error": "max_iterations must be an integer"}

    from huginn.coder.loop import CoderRunner
    from huginn.config import get_settings
    from huginn.permissions import PermissionConfig

    settings = get_settings()
    permission = PermissionConfig(auto_approve_all=auto_approve)

    get_pet_bus().publish(PetMood.THINKING, "Coder mode started", {"task": task})
    try:
        runner = CoderRunner(settings=settings, permission_config=permission)
        result = await asyncio.to_thread(
            runner.run,
            task,
            max_iterations,
        )
        get_pet_bus().publish(PetMood.SUCCESS, "Coder mode finished", {"task": task})
        return {
            "success": True,
            "final_answer": result.get("final_answer", ""),
        }
    except Exception as e:
        get_pet_bus().publish(PetMood.ERROR, f"Coder mode failed: {e}", {"task": task})
        return {"success": False, "error": str(e)}
