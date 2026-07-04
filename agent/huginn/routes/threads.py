"""Conversation thread management endpoints."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Request
from pydantic import ValidationError

from huginn.routes.schemas import CreateThreadRequest
from huginn.server_core import (
    _current_user_id,
    _state_lock,
    _threads,
    get_agent,
    get_or_create_thread,
    touch_thread,
)

router = APIRouter(tags=["threads"])


@router.get("/threads")
async def list_threads() -> dict[str, Any]:
    """List known conversation threads."""
    with _state_lock:
        threads = sorted(
            list(_threads.values()),
            key=lambda x: x.get("last_active", ""),
            reverse=True,
        )
    return {
        "threads": [
            {
                "id": t["id"],
                "label": t.get("label", t["id"]),
                "created_at": t.get("created_at", ""),
                "last_active": t.get("last_active", ""),
            }
            for t in threads
        ]
    }


@router.post("/threads")
async def create_thread(params: dict[str, Any], request: Request) -> dict[str, Any]:
    """Create a new conversation thread."""
    # Validate the request body — title length and metadata shape.
    try:
        req = CreateThreadRequest.model_validate(params)
    except ValidationError as exc:
        return {"error": f"Invalid request: {exc.errors()}"}

    thread_id = params.get("id") or uuid.uuid4().hex[:8]
    # Prefer the validated "title" field, fall back to "label" for
    # backward compat with clients that haven't migrated yet.
    label = req.title or params.get("label") or thread_id
    # Bind the thread to the authenticated caller so multi-tenant
    # deployments can isolate session data. No-op in dev / shared-key mode.
    user_id = _current_user_id(request)
    get_or_create_thread(thread_id, user_id=user_id, label=label)
    return {"id": thread_id, "label": label}


@router.get("/threads/{thread_id}")
async def get_thread(thread_id: str) -> dict[str, Any]:
    """Return metadata for a conversation thread."""
    with _state_lock:
        if thread_id in _threads:
            return {"thread_id": thread_id, **dict(_threads[thread_id])}
    return {"thread_id": thread_id, "exists": False}


@router.patch("/threads/{thread_id}")
async def rename_thread(thread_id: str, params: dict[str, Any]) -> dict[str, Any]:
    """Rename a thread."""
    with _state_lock:
        if thread_id not in _threads:
            return {"success": False, "error": "thread not found"}
        _threads[thread_id]["label"] = params.get("label", thread_id)
        return {"success": True, "label": _threads[thread_id]["label"]}


@router.delete("/threads/{thread_id}")
async def delete_thread(thread_id: str) -> dict[str, Any]:
    """Remove a thread from the registry."""
    with _state_lock:
        if thread_id in _threads:
            del _threads[thread_id]
    return {"success": True}


@router.post("/threads/{thread_id}/fork")
async def fork_thread(thread_id: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fork the conversation tree from the current position (or a given node)."""
    with _state_lock:
        if thread_id not in _threads:
            return {"success": False, "error": "thread not found"}

    agent = await get_agent()
    from_node_id = (params or {}).get("from_node_id")
    result = agent.fork_conversation(from_node_id=from_node_id)

    if result.get("success"):
        # touch_thread refreshes both last_active and last_accessed_ts so
        # the TTL sweeper treats this thread as recently used.
        touch_thread(thread_id)
    return {"thread_id": thread_id, **result}


@router.get("/threads/{thread_id}/branches")
async def list_branches(thread_id: str) -> dict[str, Any]:
    """List all branches in the conversation tree for this thread."""
    with _state_lock:
        if thread_id not in _threads:
            return {"success": False, "error": "thread not found"}

    agent = await get_agent()
    branches = agent.conversation_branches()
    return {"thread_id": thread_id, **branches}


@router.post("/threads/{thread_id}/switch-branch")
async def switch_branch(thread_id: str, params: dict[str, Any]) -> dict[str, Any]:
    """Switch the active conversation path to end at the given node."""
    with _state_lock:
        if thread_id not in _threads:
            return {"success": False, "error": "thread not found"}

    node_id = params.get("node_id")
    if not node_id:
        return {"success": False, "error": "node_id is required"}

    agent = await get_agent()
    result = agent.switch_branch(node_id)

    if result.get("success"):
        touch_thread(thread_id)
    return {"thread_id": thread_id, **result}
