"""HTTP endpoint for triggering the autonomous research loop.

The autoloop engine is normally driven from the CLI. This route lets
the Web UI start a run from a conversation and poll for progress.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from huginn.security.auth import require_api_key
from huginn.server_core import get_context

router = APIRouter(tags=["autoloop"])

logger = logging.getLogger(__name__)

# Keep background tasks alive so the GC doesn't reap them mid-run.
_pending: set[asyncio.Task] = set()

# track running autoloop tasks so /status can report back
_active_runs: dict[str, dict[str, Any]] = {}


class AutoloopStartRequest(BaseModel):
    objective: str
    max_iterations: int = 20


@router.post("/autoloop/start", dependencies=[Depends(require_api_key)])
async def start_autoloop(req: AutoloopStartRequest) -> dict[str, Any]:
    """Start an autonomous research loop and return immediately.

    The engine runs as a background task; the caller polls /tasks or
    the WebSocket progress stream for updates.
    """
    objective = req.objective.strip()
    if not objective:
        from fastapi import HTTPException

        raise HTTPException(status_code=400, detail="objective is required")

    from huginn.autoloop.engine import AutoloopEngine

    workspace = get_context().config.workspace or "."
    engine = AutoloopEngine(workspace=workspace)

    task = asyncio.create_task(engine.run(objective, max_iterations=req.max_iterations))
    _pending.add(task)
    task.add_done_callback(_pending.discard)

    _active_runs[task.get_name()] = {
        "objective": objective,
        "max_iterations": req.max_iterations,
        "task": task,
    }
    task.add_done_callback(lambda t: _active_runs.pop(t.get_name(), None))

    return {"status": "started", "objective": objective}


@router.get("/autoloop/status", dependencies=[Depends(require_api_key)])
async def autoloop_status() -> dict[str, Any]:
    """Return the status of any running autoloop tasks."""
    runs = []
    for name, info in list(_active_runs.items()):
        task = info["task"]
        runs.append({
            "id": name,
            "objective": info["objective"],
            "done": task.done(),
            "cancelled": task.cancelled(),
        })
    return {"active": len(runs), "runs": runs}
