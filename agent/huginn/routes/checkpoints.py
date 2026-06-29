"""Checkpoint / diff review endpoints."""

from __future__ import annotations

import difflib
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any

from fastapi import APIRouter

from huginn.server_core import _checkpoints, _snapshot_directory, _state_lock

router = APIRouter(tags=["checkpoints"])


@router.post("/checkpoints")
async def create_checkpoint(params: dict[str, Any]) -> dict[str, Any]:
    """Create a checkpoint of the given directory for later diff review."""
    base = Path(params.get("path", ".")).resolve()
    snapshot = _snapshot_directory(base)
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
    current = _snapshot_directory(base)
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
    current = _snapshot_directory(base)
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
