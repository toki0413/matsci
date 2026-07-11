"""Long-term memory management endpoints."""

from __future__ import annotations

import asyncio
import logging
import traceback
from typing import Any

from fastapi import APIRouter

from huginn.server_core import get_agent, get_memory_manager
from huginn.memory.types import MemoryType

router = APIRouter(tags=["memory"])

logger = logging.getLogger(__name__)


@router.get("/memory")
async def list_memories(
    category: str | None = None, tier: str | None = None, limit: int = 100
) -> dict[str, Any]:
    """List long-term memories, optionally filtered by category or tier."""
    try:
        mgr = get_memory_manager()
        if category:
            entries = mgr.longterm.list_by_category(
                category, limit=limit, alive_only=True
            )
        else:
            entries = mgr.longterm.list_all(limit=limit, alive_only=True)
        if tier:
            entries = [e for e in entries if e.get("tier") == tier]
        return {"entries": entries}
    except Exception as e:
        return {"error": str(e)}


@router.post("/memory/search")
async def search_memories(params: dict[str, Any]) -> dict[str, Any]:
    """Search long-term memory by query."""
    try:
        mgr = get_memory_manager()
        results = mgr.recall(
            query=params.get("query", ""),
            category=params.get("category"),
            tier=params.get("tier"),
            top_k=params.get("top_k", 10),
        )
        return {"results": results}
    except Exception as e:
        return {"error": str(e)}


@router.post("/memory")
async def create_memory(params: dict[str, Any]) -> dict[str, Any]:
    """Create a new memory entry."""
    try:
        mgr = get_memory_manager()
        mid = mgr.remember(
            content=params["content"],
            category=params.get("category", "fact"),
            tags=params.get("tags", []),
            importance=params.get("importance", 0.5),
            tier=params.get("tier", "mid"),
        )
        return {"memory_id": mid, "success": True}
    except Exception as e:
        return {"error": str(e), "success": False}


@router.patch("/memory/{memory_id}")
async def update_memory(memory_id: str, params: dict[str, Any]) -> dict[str, Any]:
    """Update a memory entry (content/importance/tags/tier)."""
    try:
        mgr = get_memory_manager()
        ok = mgr.longterm.update(
            memory_id,
            content=params.get("content"),
            importance=params.get("importance"),
            tags=params.get("tags"),
            tier=params.get("tier"),
        )
        return {"success": ok}
    except Exception as e:
        return {"error": str(e), "success": False}


@router.delete("/memory/{memory_id}")
async def delete_memory(memory_id: str) -> dict[str, Any]:
    """Delete a memory entry."""
    try:
        mgr = get_memory_manager()
        ok = mgr.longterm.delete(memory_id)
        return {"success": ok}
    except Exception as e:
        return {"error": str(e), "success": False}


@router.post("/memory/promote/{memory_id}")
async def promote_memory(
    memory_id: str, params: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Promote a memory to a higher tier (default long)."""
    if params is None:
        params = {}
    try:
        mgr = get_memory_manager()
        ok = mgr.longterm.promote(memory_id, target_tier=params.get("tier", "long"))
        return {"success": ok}
    except Exception as e:
        return {"error": str(e), "success": False}


@router.post("/memory/prune")
async def prune_memories(params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Prune expired and low-importance memories."""
    if params is None:
        params = {}
    try:
        mgr = get_memory_manager()
        expired = mgr.longterm.prune_expired()
        low = mgr.longterm.prune_low_importance(
            threshold=params.get("threshold", 0.2),
            older_than_days=params.get("older_than_days", 30),
        )
        return {"expired": expired, "low_importance": low}
    except Exception as e:
        return {"error": str(e)}


@router.post("/memory/sync-md")
async def sync_memory_md() -> dict[str, Any]:
    """Sync curated long-tier memories to MEMORY.md."""
    try:
        mgr = get_memory_manager()
        path = await asyncio.to_thread(mgr.sync_memory_md)
        return {"path": str(path) if path else None}
    except Exception as e:
        return {"error": str(e)}


@router.get("/memory/stats")
async def memory_stats() -> dict[str, Any]:
    """Return memory system statistics."""
    try:
        return get_memory_manager().stats()
    except Exception as e:
        return {"error": str(e)}


@router.get("/pet/memory-summary")
async def pet_memory_summary() -> dict[str, Any]:
    """Lightweight memory summary for the pet UI.

    Returns recent conversation topics, tool usage patterns,
    and memory health — enough for the pet to display context-aware
    tips and personality-driven dialogue.
    """
    try:
        mgr = get_memory_manager()
        stats = mgr.stats()

        # Recent session messages (last 5 user messages as topics)
        recent_topics: list[str] = []
        for msg in mgr.session.messages[-10:]:
            role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else "")
            content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else "")
            if role == "user" and isinstance(content, str):
                snippet = content[:80].strip()
                if snippet:
                    recent_topics.append(snippet)
        recent_topics = recent_topics[-5:]

        # Recent tool usage (last 5 tool calls)
        recent_tools: list[dict[str, Any]] = []
        for tc in mgr.session.tool_calls[-5:]:
            name = getattr(tc, "tool_name", None) or (tc.get("tool_name") if isinstance(tc, dict) else "")
            if name:
                recent_tools.append({"tool": name})

        # Top long-term memory categories
        top_categories: list[str] = []
        try:
            all_entries = mgr.longterm.list_all(limit=50, alive_only=True)
            cat_counts: dict[str, int] = {}
            for e in all_entries:
                cat = e.get("category", "fact")
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
            top_categories = sorted(cat_counts, key=cat_counts.get, reverse=True)[:5]
        except Exception:
            pass

        return {
            "session_topics": recent_topics,
            "tool_calls_recent": recent_tools,
            "tool_calls_total": stats.get("session_tool_calls", 0),
            "memory_entries": stats.get("longterm_entries", 0),
            "top_categories": top_categories,
            "session_messages": stats.get("session_messages", 0),
        }
    except Exception as e:
        logger.warning("pet memory summary failed", exc_info=True)
        return {"error": str(e)}


@router.post("/memory/maintenance")
async def memory_maintenance(params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run long-term memory decay, prune, and deduplication."""
    try:
        agent = await get_agent()
        p = params or {}
        summary = await asyncio.to_thread(
            agent.memory.maintenance,
            prune_threshold=p.get("prune_threshold", 0.15),
            deduplicate=p.get("deduplicate", True),
        )
        return {"success": True, "summary": summary}
    except Exception as e:
        logger.error("unexpected error", exc_info=True)
        return {"success": False, "error": str(e)}


@router.post("/memory/lint")
async def memory_lint(params: dict[str, Any] | None = None) -> dict[str, Any]:
    """LLM Wiki Lint: knowledge base health check."""
    try:
        agent = await get_agent()
        p = params or {}
        report = await asyncio.to_thread(
            agent.memory.longterm.lint,
            limit=p.get("limit", 100),
        )
        return {"success": True, "report": report}
    except Exception as e:
        logger.error("lint error", exc_info=True)
        return {"success": False, "error": str(e)}


# ── typed memory: filesystem-based topic notes ──────────────────────


@router.get("/memory/typed")
async def list_typed_memory(
    memory_type: str, topic: str | None = None
) -> dict[str, Any]:
    """Recall topic-organized markdown notes by memory type."""
    try:
        mt = MemoryType(memory_type)
    except ValueError:
        return {"error": f"invalid memory_type: {memory_type}"}
    try:
        mgr = get_memory_manager()
        results = mgr.recall_typed(mt, topic=topic)
        return {"entries": results}
    except Exception as e:
        return {"error": str(e)}


@router.post("/memory/typed")
async def create_typed_memory(params: dict[str, Any]) -> dict[str, Any]:
    """Store a topic-organized markdown note."""
    try:
        mt = MemoryType(params["memory_type"])
    except (KeyError, ValueError) as e:
        return {"error": f"invalid memory_type: {e}"}
    try:
        mgr = get_memory_manager()
        path = mgr.store_typed_memory(
            mt, params["topic"], params["content"]
        )
        return {"path": str(path), "success": True}
    except Exception as e:
        return {"error": str(e), "success": False}


@router.get("/memory/typed/index")
async def typed_memory_index() -> dict[str, Any]:
    """Return a text index of all topic files."""
    try:
        mgr = get_memory_manager()
        return {"index": mgr.get_memory_index()}
    except Exception as e:
        return {"error": str(e)}
