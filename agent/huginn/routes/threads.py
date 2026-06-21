"""Conversation thread management endpoints."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter

from huginn.server_core import _threads

router = APIRouter(tags=["threads"])


@router.get("/threads")
async def list_threads() -> dict[str, Any]:
    """List known conversation threads."""
    return {
        "threads": [
            {
                "id": t["id"],
                "label": t.get("label", t["id"]),
                "created_at": t.get("created_at", ""),
                "last_active": t.get("last_active", ""),
            }
            for t in sorted(
                _threads.values(), key=lambda x: x.get("last_active", ""), reverse=True
            )
        ]
    }


@router.post("/threads")
async def create_thread(params: dict[str, Any]) -> dict[str, Any]:
    """Create a new conversation thread."""
    thread_id = params.get("id") or uuid.uuid4().hex[:8]
    label = params.get("label") or thread_id
    _threads[thread_id] = {
        "id": thread_id,
        "label": label,
        "created_at": uuid.uuid4().hex,
        "last_active": uuid.uuid4().hex,
    }
    return {"id": thread_id, "label": label}


@router.get("/threads/{thread_id}")
async def get_thread(thread_id: str) -> dict[str, Any]:
    """Return metadata for a conversation thread."""
    if thread_id in _threads:
        return {"thread_id": thread_id, **dict(_threads[thread_id])}
    return {"thread_id": thread_id, "exists": False}


@router.patch("/threads/{thread_id}")
async def rename_thread(thread_id: str, params: dict[str, Any]) -> dict[str, Any]:
    """Rename a thread."""
    if thread_id not in _threads:
        return {"success": False, "error": "thread not found"}
    _threads[thread_id]["label"] = params.get("label", thread_id)
    return {"success": True, "label": _threads[thread_id]["label"]}


@router.delete("/threads/{thread_id}")
async def delete_thread(thread_id: str) -> dict[str, Any]:
    """Remove a thread from the registry."""
    if thread_id in _threads:
        del _threads[thread_id]
    return {"success": True}
