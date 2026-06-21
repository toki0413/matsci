"""Codebase semantic search endpoints."""

from __future__ import annotations

import asyncio
import traceback
from typing import Any

from fastapi import APIRouter

from huginn.server_core import get_context

router = APIRouter(tags=["codebase"])


@router.get("/codebase")
async def codebase_status() -> dict[str, Any]:
    """Return codebase index status."""
    if get_context().codebase is None:
        return {"available": False, "error": "Codebase index not initialized"}
    try:
        return get_context().codebase.status()
    except Exception as e:
        return {"available": False, "error": str(e)}


@router.post("/codebase/index")
async def codebase_index() -> dict[str, Any]:
    """Re-index the workspace codebase."""
    if get_context().codebase is None:
        return {"success": False, "error": "Codebase index not initialized"}
    try:
        return {
            "success": True,
            **await asyncio.to_thread(get_context().codebase.index_workspace),
        }
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@router.post("/codebase/search")
async def codebase_search(params: dict[str, Any]) -> dict[str, Any]:
    """Search the codebase index."""
    if get_context().codebase is None:
        return {"results": [], "error": "Codebase index not initialized"}
    try:
        query = params.get("query", "")
        top_k = int(params.get("top_k", 5))
        results = await asyncio.to_thread(get_context().codebase.search, query, top_k)
        return {"results": results}
    except Exception as e:
        traceback.print_exc()
        return {"results": [], "error": str(e)}
