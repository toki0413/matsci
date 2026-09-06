"""HTTP endpoints for the provenance registry.

Thin read/delete wrappers over ProvenanceRegistry so the UI can browse
recent file outputs, search, trace lineage, and prune old entries.
The registry is a process-wide singleton (ProvenanceRegistry.shared()).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from huginn.provenance.registry import ProvenanceRegistry

router = APIRouter(prefix="/provenance", tags=["provenance"])


def _registry() -> ProvenanceRegistry:
    return ProvenanceRegistry.shared()


def _registry_or_503() -> ProvenanceRegistry:
    """Return the shared registry, or 503 if it cannot be initialised."""
    try:
        return ProvenanceRegistry.shared()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"provenance registry unavailable: {exc}",
        ) from exc


@router.get("")
@router.get("/")
async def root(n: int = 20):
    """Root alias for /provenance/recent — convenience for UI."""
    return await recent(n)


@router.get("/recent")
async def recent(n: int = 20):
    try:
        entries = _registry().recent(n)
        return {"success": True, "data": [e.to_dict() for e in entries]}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@router.get("/count")
async def count():
    try:
        return {"success": True, "data": _registry().count()}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@router.get("/search")
async def search(q: str):
    try:
        # query() already returns scored list[dict] (serialized entries)
        return {"success": True, "data": _registry().query(q)}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@router.get("/by-tool")
async def by_tool(tool: str):
    try:
        entries = _registry().find_by_tool(tool)
        return {"success": True, "data": [e.to_dict() for e in entries]}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@router.get("/by-format")
async def by_format(fmt: str):
    try:
        entries = _registry().find_by_format(fmt)
        return {"success": True, "data": [e.to_dict() for e in entries]}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@router.get("/lineage")
async def lineage(path: str, depth: int = 5):
    try:
        entries = _registry().get_lineage(path, depth)
        return {"success": True, "data": [e.to_dict() for e in entries]}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@router.get("/dag")
async def dag(n: int = 50):
    """Return nodes + edges for provenance DAG visualization."""
    try:
        entries = _registry().recent(n)
        nodes = []
        edges = []
        seen = set()
        for e in entries:
            d = e.to_dict()
            node_id = d.get("path") or d.get("file_id") or d.get("id")
            if node_id and node_id not in seen:
                seen.add(node_id)
                nodes.append({
                    "id": node_id,
                    "label": d.get("filename") or node_id.rsplit("/", 1)[-1],
                    "tool": d.get("tool"),
                    "format": d.get("format"),
                    "timestamp": d.get("timestamp"),
                })
            # trace lineage edges
            parent = d.get("derived_from") or d.get("parent_path")
            if parent and node_id and parent != node_id:
                edges.append({"source": parent, "target": node_id})
        return {"success": True, "data": {"nodes": nodes, "edges": edges}}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@router.delete("/cleanup")
async def cleanup(days: int = 30):
    try:
        deleted = _registry().cleanup_old(days)
        return {"success": True, "data": {"deleted": deleted}}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


# ── Event-sourcing read/replay API ──────────────────────────────
# These expose ProvenanceRegistry's event-sourcing surface so the
# event store is no longer an orphan module with no callers.


@router.get("/events")
async def events(since_id: int = 0, limit: int = 100, tool: str = ""):
    """Event-sourcing tail: events after ``since_id`` in chronological order.

    ``tool`` filters by producer; an empty value means no filter.
    """
    try:
        reg = _registry_or_503()
        entries = reg.get_events(since_id=since_id, limit=limit, tool=(tool or None))
        return {"success": True, "data": [e.to_dict() for e in entries]}
    except HTTPException:
        raise
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@router.get("/events/{event_id}")
async def event(event_id: int):
    """Fetch a single event by id. 404 when the id does not exist."""
    try:
        reg = _registry_or_503()
        entry = reg.get_event(event_id)
        if entry is None:
            raise HTTPException(
                status_code=404, detail=f"event {event_id} not found"
            )
        return {"success": True, "data": entry.to_dict()}
    except HTTPException:
        raise
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@router.get("/replay/{target_id}")
async def replay(target_id: int):
    """Replay every event from the beginning up to ``target_id`` (inclusive)."""
    try:
        reg = _registry_or_503()
        entries = reg.replay_to(target_id)
        return {"success": True, "data": [e.to_dict() for e in entries]}
    except HTTPException:
        raise
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@router.get("/version")
async def version():
    """Current version clock (max event id) for optimistic concurrency."""
    try:
        reg = _registry_or_503()
        return {"success": True, "data": reg.current_version()}
    except HTTPException:
        raise
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@router.get("/by_seed")
async def by_seed(seed: int):
    """Find events produced with a given random seed (reproducibility check)."""
    try:
        reg = _registry_or_503()
        entries = reg.find_by_seed(seed)
        return {"success": True, "data": [e.to_dict() for e in entries]}
    except HTTPException:
        raise
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@router.get("/by-property")
async def by_property(key: str, value: str = ""):
    """Find entries by ``key_properties[key] == value`` (empty value → any)."""
    try:
        reg = _registry_or_503()
        # value 是查询参数, 空串表示 "只要包含这个 key 就行", 对齐
        # registry.find_by_property 的 value=None 语义.
        entries = reg.find_by_property(key, value if value else None)
        return {"success": True, "data": [e.to_dict() for e in entries]}
    except HTTPException:
        raise
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@router.get("/skills-snapshot")
async def skills_snapshot():
    """Latest skill-manifest snapshot written by ``snapshot_skills``.

    Pairs with the write side in ``plugins/science_skills_bridge.py`` so the
    snapshot is no longer write-only — agents/UI can read it back to compare
    the live skill set against the last recorded version.
    """
    try:
        reg = _registry_or_503()
        skills = reg.load_skills_snapshot()
        return {"success": True, "data": skills}
    except HTTPException:
        raise
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@router.post("/revert/{version}")
async def revert(version: int):
    """Roll the registry back to ``version`` (write op → POST).

    Returns the list of file paths affected by the revert.
    """
    try:
        reg = _registry_or_503()
        reverted_paths = reg.revert_to_version(version)
        return {
            "success": True,
            "data": {"version": version, "reverted_paths": reverted_paths},
        }
    except HTTPException:
        raise
    except Exception as exc:
        return {"success": False, "error": str(exc)}
