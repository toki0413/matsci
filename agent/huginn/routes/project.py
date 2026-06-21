"""Project context file endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from huginn.config import HuginnConfig
from huginn.project_context import (
    context_source,
    load_project_context,
    project_context_path,
    save_project_context,
)
from huginn.server_core import get_context

router = APIRouter(tags=["project"])


@router.get("/project-context")
async def get_project_context() -> dict[str, Any]:
    """Return the current project context file content and source."""
    cfg = HuginnConfig.from_env()
    return {
        "source": context_source(cfg.workspace),
        "path": str(project_context_path(cfg.workspace)),
        "content": load_project_context(cfg.workspace),
    }


@router.post("/project-context")
async def update_project_context(params: dict[str, Any]) -> dict[str, Any]:
    """Update the `.huginn.md` project context file."""
    cfg = HuginnConfig.from_env()
    content = params.get("content", "")
    try:
        result = save_project_context(cfg.workspace, content)
        get_context().agent = None  # force re-init so new context is loaded
        return {"success": True, **result}
    except Exception as e:
        return {"success": False, "error": str(e)}
