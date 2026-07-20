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


@router.get("/memory/retrieve")
async def retrieve_memories(q: str = "", limit: int = 10) -> dict[str, Any]:
    """GET alias for memory search — convenience for simple lookups."""
    try:
        mgr = get_memory_manager()
        results = mgr.recall(query=q, top_k=limit)
        return {"results": results}
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


@router.get("/memory/layers")
async def memory_layers() -> dict[str, Any]:
    """聚合 4 层 memory (WM/EM/SM/PM) 状态, 给前端 Memory 层级面板用.

    每层独立 try/except, 单层失败不阻塞其他层. 零新依赖, 全部复用
    现有 manager / longterm / kb / kg / load_stable_principles.
    ponytail: 不包 LayerAggregator 抽象, 直接调现成方法.
    """
    from huginn.server_core import get_context
    from pathlib import Path

    result: dict[str, Any] = {}

    # ── WM: 当前 session 的 token 占用 / summaries / last_summarize_at ──
    try:
        mgr = get_memory_manager()
        session = mgr.session
        result["wm"] = {
            "token_used": session._estimate_tokens(),
            "token_budget": session.token_budget,
            "messages_count": len(session.messages),
            "tool_calls_count": len(session.tool_calls),
            "summaries_count": len(session.summaries),
            "last_summarize_at": session.last_summarize_at,
            "extreme_dispatch": (
                __import__("os").environ.get("HUGINN_EXTREME_DISPATCH", "0").lower()
                in ("1", "true")
            ),
        }
    except Exception as e:
        result["wm"] = {"error": str(e), "available": False}

    # ── EM: SQLite memories 表 — 总数 + tier 分布 + 最近 10 条 episode ──
    try:
        mgr = get_memory_manager()
        stats = mgr.stats()
        tier_counts = stats.get("tier_counts", {})
        recent_episodes_raw = mgr.longterm.list_by_category(
            "episode", limit=10, alive_only=True
        )
        recent_episodes = [
            {
                "id": str(ep.get("id", "")),
                "content": (ep.get("content") or "")[:200],
                "last_accessed": ep.get("last_accessed"),
                "importance": ep.get("importance"),
                "source": ep.get("source"),
            }
            for ep in recent_episodes_raw
        ]
        result["em"] = {
            "total_entries": stats.get("longterm_entries", 0),
            "tier_counts": tier_counts,
            "recent_episodes": recent_episodes,
        }
    except Exception as e:
        result["em"] = {"error": str(e), "available": False}

    # ── SM: KB chunks + KG 节点 + 最近写入的 trajectory_pattern ──
    # KB 走 ServerContext, KG 临时实例化 (YAGNI — 只此端点用)
    try:
        ctx = get_context()
        kb = getattr(ctx, "kb", None)
        kb_chunks = kb.count() if kb is not None else 0
        # trajectory_pattern 也存在 KB 里 (metadata source=trajectory_pattern)
        recent_patterns: list[dict[str, Any]] = []
        top_patterns_by_confidence: list[dict[str, Any]] = []
        if kb is not None:
            try:
                data = kb.collection.get(
                    where={"source": "trajectory_pattern"},
                    include=["metadatas", "documents"],
                )
                metas = data.get("metadatas") or []
                docs = data.get("documents") or []
                ids = data.get("ids") or []
                rows = list(zip(ids, metas, docs))
                # 最近写入: 按 run_id 字符串降序 (run_id 通常是 timestamp 格式)
                def _run_id(r):
                    return str((r[1] or {}).get("run_id") or "")
                recent_rows = sorted(rows, key=_run_id, reverse=True)[:5]
                for _id, meta, doc in recent_rows:
                    recent_patterns.append({
                        "doc_id": str(_id),
                        "task_pattern": (meta or {}).get("task_pattern", ""),
                        "run_id": (meta or {}).get("run_id", ""),
                        "objective": (meta or {}).get("objective", ""),
                        "confidence": _safe_float((meta or {}).get("confidence"), 0.5),
                        "doc_preview": (doc or "")[:200],
                    })
                # top-5 by confidence
                def _conf(r):
                    return _safe_float((r[1] or {}).get("confidence"), 0.0)
                top_rows = sorted(rows, key=_conf, reverse=True)[:5]
                for _id, meta, doc in top_rows:
                    top_patterns_by_confidence.append({
                        "doc_id": str(_id),
                        "task_pattern": (meta or {}).get("task_pattern", ""),
                        "confidence": _safe_float((meta or {}).get("confidence"), 0.5),
                        "objective": (meta or {}).get("objective", ""),
                    })
            except Exception:
                pass  # KB collection 查询失败不阻塞, recent_patterns 留空
        # KG 临时实例化
        kg_stats: dict[str, Any] = {"nodes": 0, "edges": 0, "node_types": {}}
        try:
            from huginn.kg.graph import ProjectKnowledgeGraph
            workspace = Path(getattr(ctx, "workspace", ".") or ".")
            kg = ProjectKnowledgeGraph(workspace / ".huginn")
            kg_stats = kg.stats()
        except Exception:
            pass  # KG 文件不存在或损坏不阻塞
        result["sm"] = {
            "kb_chunks": kb_chunks,
            "kg_nodes": kg_stats.get("nodes", 0),
            "kg_edges": kg_stats.get("edges", 0),
            "kg_node_types": kg_stats.get("node_types", {}),
            "recent_patterns": recent_patterns,
        }
    except Exception as e:
        result["sm"] = {"error": str(e), "available": False}

    # ── PM: stable_principles (JSONL) + top_patterns_by_confidence (从 SM 查) ──
    try:
        from huginn.memory.longterm import load_stable_principles
        principles = load_stable_principles()
        result["pm"] = {
            "stable_principles_count": len(principles),
            "stable_principles_preview": principles[:5],
            # top_patterns_by_confidence 复用 SM 那次 collection.get 的结果
            "top_patterns_by_confidence": top_patterns_by_confidence,
        }
    except Exception as e:
        result["pm"] = {"error": str(e), "available": False}

    return result


def _safe_float(v: Any, default: float = 0.0) -> float:
    """容错把 ChromaDB metadata 的 string confidence 转 float."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


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
