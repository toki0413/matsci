"""HTTP endpoint for triggering the autonomous research loop.

The autoloop engine is normally driven from the CLI. This route lets
the Web UI start a run from a conversation and poll for progress.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from huginn.interaction.progress import get_progress_tracker
from huginn.security.auth import require_api_key, require_capability
from huginn.server_core import get_context

# G32: autoloop 在后台跑自主循环 (调工具/写文件), 走 execute capability.
router = APIRouter(
    tags=["autoloop"],
    dependencies=[Depends(require_api_key), Depends(require_capability("execute"))],
)

logger = logging.getLogger(__name__)

# Keep background tasks alive so the GC doesn't reap them mid-run.
_pending: set[asyncio.Task] = set()

# track running autoloop tasks so /status can report back
_active_runs: dict[str, dict[str, Any]] = {}


class AutoloopStartRequest(BaseModel):
    objective: str
    max_iterations: int = 20


def _make_done_callback(progress_task_id: str, run_key: str):
    """task.add_done_callback: 拿结果标 complete/fail, 推到 /tasks/stream SSE.

    ponytail: 复用 ProgressTracker 的 SSE 通道, 不另起 WS 广播.
    """

    def _cb(t: asyncio.Task) -> None:
        try:
            _active_runs.pop(run_key, None)
        except Exception:
            pass
        tracker = get_progress_tracker()
        if t.cancelled():
            tracker.fail(progress_task_id, "autoloop cancelled")
            return
        exc = t.exception()
        if exc is not None:
            tracker.fail(progress_task_id, f"autoloop error: {exc}")
            return
        # result 是 AutoloopResult; 取关键字段塞进 result
        try:
            res = t.result()
            summary = {
                "success": getattr(res, "success", True),
                "objective": getattr(res, "objective", ""),
                "run_id": getattr(res, "run_id", ""),
                "phases": len(getattr(res, "phases", []) or []),
                "goal_achieved": getattr(res, "goal_achieved", None),
            }
            tracker.complete(progress_task_id, result=summary)
        except Exception as e:
            logger.warning("autoloop done callback result parse failed: %s", e)
            tracker.complete(progress_task_id, result={"success": True})

    return _cb


@router.post("/autoloop/start")
async def start_autoloop(req: AutoloopStartRequest) -> dict[str, Any]:
    """Start an autonomous research loop and return immediately.

    The engine runs as a background task; the caller polls /tasks or
    the WebSocket progress stream for updates. 完成时通过 ProgressTracker
    SSE (/tasks/stream) 推送 autoloop_done 事件.
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

    # 登记一个顶层 progress task, 完成时 done_callback 标 complete/fail.
    # 这样 /tasks/stream SSE 客户端能收到明确的 done 事件, 而不是靠轮询 task.done().
    run_key = task.get_name()
    progress_task_id = f"autoloop:{run_key}:{uuid.uuid4().hex[:8]}"
    tracker = get_progress_tracker()
    tracker.start_task(
        task_id=progress_task_id,
        description=f"autoloop: {objective[:120]}",
        total_steps=req.max_iterations,
        stage_labels=["perceive", "hypothesize", "plan", "execute", "validate", "learn"],
        engine_kind="autoloop",
        metadata={"objective": objective, "run_key": run_key},
    )
    _active_runs[run_key] = {
        "objective": objective,
        "max_iterations": req.max_iterations,
        "task": task,
        "progress_task_id": progress_task_id,
    }
    task.add_done_callback(_make_done_callback(progress_task_id, run_key))

    return {
        "status": "started",
        "objective": objective,
        "progress_task_id": progress_task_id,
    }


@router.get("/autoloop/status")
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
