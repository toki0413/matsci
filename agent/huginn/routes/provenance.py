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


@router.delete("/cleanup")
async def cleanup(days: int = 30):
    try:
        deleted = _registry().cleanup_old(days)
        return {"success": True, "data": {"deleted": deleted}}
    except Exception as exc:
        return {"success": False, "error": str(exc)}
