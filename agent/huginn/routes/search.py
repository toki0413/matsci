"""Global search endpoint — aggregates results from multiple sources.

Supports searching across threads, memory, knowledge base, and provenance
in a single unified query. Results include type metadata for frontend
to render appropriate icons and actions.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, Request

from huginn.routes.threads import _check_thread_owner
from huginn.server_core import (
    _threads,
    _state_lock,
    get_agent,
    get_context,
    get_memory_manager,
)
from huginn.provenance.registry import ProvenanceRegistry

router = APIRouter(tags=["search"])

logger = logging.getLogger(__name__)

SEARCH_SOURCE_TYPES = ["thread", "memory", "knowledge", "provenance"]


class GlobalSearchResult:
    """Unified search result format across all sources."""

    def __init__(
        self,
        type: str,
        id: str,
        title: str,
        snippet: str,
        score: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ):
        self.type = type
        self.id = id
        self.title = title
        self.snippet = snippet
        self.score = score
        self.metadata = metadata or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "id": self.id,
            "title": self.title,
            "snippet": self.snippet,
            "score": self.score,
            "metadata": self.metadata,
        }


def _extract_snippet(text: str, query: str, max_len: int = 150) -> str:
    """Extract a relevant snippet from text around query matches."""
    text = str(text).strip()
    if not text:
        return ""

    query_lower = query.lower()
    text_lower = text.lower()
    idx = text_lower.find(query_lower)

    if idx >= 0:
        start = max(0, idx - 30)
        end = min(len(text), idx + len(query) + 120)
        snippet = text[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."
        return snippet

    return text[:max_len] + ("..." if len(text) > max_len else "")


async def _search_threads(query: str, request: Request) -> list[GlobalSearchResult]:
    """Search thread labels and recent messages."""
    results: list[GlobalSearchResult] = []
    user_id = None
    try:
        from huginn.server_core import _current_user_id

        user_id = _current_user_id(request)
    except Exception:
        pass

    with _state_lock:
        for thread_id, thread in _threads.items():
            if user_id not in ("dev", "shared", None):
                owner = thread.get("user_id")
                if owner and owner != user_id:
                    continue

            label = thread.get("label", "")
            lower_query = query.lower()
            match_score = 0.0

            if lower_query in label.lower():
                match_score += 0.8

            recent_messages = []
            try:
                agent = await get_agent()
                graph = agent.build_graph()
                if graph:
                    config = {"configurable": {"thread_id": thread_id}}
                    snapshot = graph.get_state(config)
                    raw_msgs = snapshot.values.get("messages", []) if snapshot else []
                    for msg in raw_msgs[-10:]:
                        content = getattr(msg, "content", "") or ""
                        if isinstance(content, str) and lower_query in content.lower():
                            match_score += 0.3
                            recent_messages.append(content)
            except Exception:
                pass

            if match_score > 0:
                if recent_messages:
                    snippet = _extract_snippet(recent_messages[-1], query)
                else:
                    snippet = _extract_snippet(label, query)

                results.append(
                    GlobalSearchResult(
                        type="thread",
                        id=thread_id,
                        title=label or thread_id,
                        snippet=snippet,
                        score=match_score,
                        metadata={
                            "created_at": thread.get("created_at", ""),
                            "last_active": thread.get("last_active", ""),
                        },
                    )
                )

    return results


async def _search_memory(query: str) -> list[GlobalSearchResult]:
    """Search long-term memory entries."""
    results: list[GlobalSearchResult] = []
    try:
        mgr = get_memory_manager()
        mem_results = mgr.recall(query=query, top_k=10)
        for entry in mem_results:
            content = entry.get("content", "")
            score = entry.get("score", 0.5)
            results.append(
                GlobalSearchResult(
                    type="memory",
                    id=entry.get("id", ""),
                    title=entry.get("category", "fact"),
                    snippet=_extract_snippet(content, query),
                    score=score,
                    metadata={
                        "category": entry.get("category", ""),
                        "tier": entry.get("tier", ""),
                        "importance": entry.get("importance", 0),
                        "created_at": entry.get("created_at", ""),
                    },
                )
            )
    except Exception as e:
        logger.debug("memory search failed: %s", e)

    return results


async def _search_knowledge(query: str) -> list[GlobalSearchResult]:
    """Search knowledge base documents."""
    results: list[GlobalSearchResult] = []
    kb = get_context().kb
    if kb is None:
        return results

    try:
        chunks = kb.query(query, top_k=10)
        for chunk in chunks:
            doc_id = chunk.get("doc_id", "")
            filename = chunk.get("filename", "")
            content = chunk.get("content", "")
            score = chunk.get("score", 0.5)

            results.append(
                GlobalSearchResult(
                    type="knowledge",
                    id=doc_id,
                    title=filename or doc_id,
                    snippet=_extract_snippet(content, query),
                    score=score,
                    metadata={
                        "doc_id": doc_id,
                        "filename": filename,
                    },
                )
            )
    except Exception as e:
        logger.debug("knowledge search failed: %s", e)

    return results


async def _search_provenance(query: str) -> list[GlobalSearchResult]:
    """Search provenance registry (files/outputs)."""
    results: list[GlobalSearchResult] = []
    try:
        registry = ProvenanceRegistry.shared()
        entries = registry.query(query)
        for entry in entries[:10]:
            path = entry.get("path", "")
            filename = entry.get("filename", "")
            description = entry.get("description", "") or entry.get("output", "") or ""
            score = entry.get("score", 0.5)

            results.append(
                GlobalSearchResult(
                    type="provenance",
                    id=path or entry.get("id", ""),
                    title=filename or path.rsplit("/", 1)[-1] if path else "untitled",
                    snippet=_extract_snippet(description, query),
                    score=score,
                    metadata={
                        "path": path,
                        "tool": entry.get("tool", ""),
                        "format": entry.get("format", ""),
                        "timestamp": entry.get("timestamp", ""),
                    },
                )
            )
    except Exception as e:
        logger.debug("provenance search failed: %s", e)

    return results


@router.get("/search/global")
async def global_search(
    query: str,
    limit: int = 20,
    sources: str | None = None,
    request: Request = None,
) -> dict[str, Any]:
    """Search across threads, memory, knowledge, and provenance.

    Args:
        query: Search term
        limit: Maximum number of results to return
        sources: Comma-separated list of sources to include.
                 Valid values: thread, memory, knowledge, provenance.
                 If omitted, searches all sources.
    """
    if not query or len(query.strip()) < 2:
        return {"results": [], "error": "query must be at least 2 characters"}

    query = query.strip()

    selected_sources = SEARCH_SOURCE_TYPES
    if sources:
        selected_sources = [s.strip() for s in sources.split(",") if s.strip() in SEARCH_SOURCE_TYPES]
        if not selected_sources:
            return {"results": [], "error": f"invalid sources; valid: {','.join(SEARCH_SOURCE_TYPES)}"}

    tasks = []
    if "thread" in selected_sources:
        tasks.append(_search_threads(query, request))
    if "memory" in selected_sources:
        tasks.append(_search_memory(query))
    if "knowledge" in selected_sources:
        tasks.append(_search_knowledge(query))
    if "provenance" in selected_sources:
        tasks.append(_search_provenance(query))

    all_results: list[GlobalSearchResult] = []
    if tasks:
        results_list = await asyncio.gather(*tasks)
        for results in results_list:
            all_results.extend(results)

    all_results.sort(key=lambda r: r.score, reverse=True)
    all_results = all_results[:limit]

    return {
        "results": [r.to_dict() for r in all_results],
        "total": len(all_results),
        "sources": selected_sources,
    }
