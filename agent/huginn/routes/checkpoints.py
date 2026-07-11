"""Checkpoint / diff review endpoints."""

from __future__ import annotations

import asyncio
import difflib
import logging
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from huginn.server_core import _checkpoints, _snapshot_directory, _state_lock

logger = logging.getLogger(__name__)

router = APIRouter(tags=["checkpoints"])

# Maximum allowed path depth to prevent excessive recursion.
_MAX_PATH_DEPTH = 20


def _validate_workspace_path(raw_path: str) -> Path:
    """Ensure the requested path stays within the workspace boundary.

    Prevents path traversal attacks where an attacker could read or
    write arbitrary system files by passing paths like ``/etc`` or
    ``~/.ssh``.
    """
    from huginn.server_core import get_context

    try:
        workspace = get_context().config.workspace
        workspace_resolved = Path(workspace).resolve()
    except Exception:
        # Fallback: use current working directory
        workspace_resolved = Path.cwd()

    base = Path(raw_path).resolve()

    # Check the resolved path is within the workspace
    try:
        base.relative_to(workspace_resolved)
    except ValueError:
        raise HTTPException(
            status_code=403,
            detail=f"Path '{raw_path}' is outside the workspace boundary",
        )

    # Prevent excessive depth
    rel = base.relative_to(workspace_resolved)
    if len(rel.parts) > _MAX_PATH_DEPTH:
        raise HTTPException(
            status_code=400,
            detail="Path exceeds maximum allowed depth",
        )

    return base


@router.post("/checkpoints")
async def create_checkpoint(params: dict[str, Any]) -> dict[str, Any]:
    """Create a checkpoint of the given directory for later diff review."""
    base = _validate_workspace_path(params.get("path", "."))
    snapshot = await asyncio.to_thread(_snapshot_directory, base)
    cp_id = uuid.uuid4().hex[:8]
    with _state_lock:
        _checkpoints[cp_id] = (base, snapshot)
    return {"id": cp_id, "base": str(base), "files": len(snapshot)}


@router.get("/checkpoints/{cp_id}")
async def get_checkpoint(cp_id: str) -> dict[str, Any]:
    with _state_lock:
        if cp_id not in _checkpoints:
            return {"error": "checkpoint not found"}
        base, snapshot = _checkpoints[cp_id]
    return {"id": cp_id, "base": str(base), "files": list(snapshot.keys())}


@router.get("/checkpoints/{cp_id}/diff")
async def checkpoint_diff(cp_id: str) -> dict[str, Any]:
    with _state_lock:
        if cp_id not in _checkpoints:
            return {"error": "checkpoint not found"}
        base, snapshot = _checkpoints[cp_id]
    current = await asyncio.to_thread(_snapshot_directory, base)
    diffs = []
    all_files = set(snapshot.keys()) | set(current.keys())
    for rel in sorted(all_files):
        old = snapshot.get(rel, "")
        new = current.get(rel, "")
        if old == new:
            continue
        status = (
            "added"
            if rel not in snapshot
            else "deleted" if rel not in current else "modified"
        )
        diff_text = "\n".join(
            difflib.unified_diff(
                old.splitlines(),
                new.splitlines(),
                fromfile=f"a/{rel}",
                tofile=f"b/{rel}",
                lineterm="",
            )
        )
        diffs.append(
            {
                "path": rel,
                "status": status,
                "diff": diff_text,
                "old": old,
                "new": new,
            }
        )
    return {"id": cp_id, "base": str(base), "diffs": diffs}


@router.post("/checkpoints/{cp_id}/accept")
async def accept_checkpoint(cp_id: str) -> dict[str, Any]:
    with _state_lock:
        if cp_id not in _checkpoints:
            return {"error": "checkpoint not found"}
        del _checkpoints[cp_id]
    return {"success": True}


@router.post("/checkpoints/{cp_id}/reject")
async def reject_checkpoint(cp_id: str) -> dict[str, Any]:
    with _state_lock:
        if cp_id not in _checkpoints:
            return {"error": "checkpoint not found"}
        base, snapshot = _checkpoints[cp_id]
        del _checkpoints[cp_id]
    current = await asyncio.to_thread(_snapshot_directory, base)
    for rel, content in snapshot.items():
        if rel in current:
            path = base / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    # Remove files that were added after the checkpoint
    for rel in current:
        if rel not in snapshot:
            path = base / rel
            with suppress(Exception):
                path.unlink()
    return {"success": True}
