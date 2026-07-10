"""HTTP endpoints for the provenance registry.

Thin read/delete wrappers over ProvenanceRegistry so the UI can browse
recent file outputs, search, trace lineage, and prune old entries.
The registry is a process-wide singleton (ProvenanceRegistry.shared()).
"""

from __future__ import annotations

from fastapi import APIRouter

from huginn.provenance.registry import ProvenanceRegistry

router = APIRouter(prefix="/provenance", tags=["provenance"])


def _registry() -> ProvenanceRegistry:
    return ProvenanceRegistry.shared()


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
