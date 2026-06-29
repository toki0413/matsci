"""Knowledge base / RAG and export endpoints."""

from __future__ import annotations

import json
import tempfile
import traceback
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import FileResponse

from huginn.security.auth import require_api_key
from huginn.server_core import get_context

router = APIRouter(tags=["knowledge"])

_MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB


@router.post("/knowledge/upload")
async def upload_knowledge(file: UploadFile = File(...)) -> dict[str, Any]:
    """Upload a document to the private knowledge base."""
    if get_context().kb is None:
        return {"error": "Knowledge base is not available"}
    try:
        content = await file.read()
        if len(content) > _MAX_UPLOAD_BYTES:
            return {
                "success": False,
                "error": f"File too large ({len(content)} bytes, max {_MAX_UPLOAD_BYTES})",
            }
        result = get_context().kb.add_document(file.filename or "unnamed", content)
        return {"success": True, "document": result}
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@router.get("/knowledge")
async def list_knowledge() -> dict[str, Any]:
    """List documents in the knowledge base."""
    if get_context().kb is None:
        return {"documents": [], "available": False}
    try:
        return {
            "documents": get_context().kb.list_documents(),
            "count": get_context().kb.count(),
            "available": True,
        }
    except Exception as e:
        return {"documents": [], "error": str(e), "available": False}


@router.delete("/knowledge/{doc_id}")
async def delete_knowledge(doc_id: str) -> dict[str, Any]:
    """Remove a document from the knowledge base."""
    if get_context().kb is None:
        return {"success": False, "error": "Knowledge base is not available"}
    try:
        deleted = get_context().kb.delete_document(doc_id)
        return {"success": True, "deleted": deleted}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/export", dependencies=[Depends(require_api_key)])
async def export_data(
    source: str = "audit",
    fmt: str = "json",
) -> Any:
    """Export Huginn records as a downloadable file."""
    from huginn.export_manager import ExportManager

    manager = ExportManager(get_context().config.workspace)
    suffix = "md" if fmt == "markdown" else fmt
    output = Path(tempfile.gettempdir()) / f"huginn_export_{source}.{suffix}"
    result = manager.export(source=source, output_path=output, fmt=fmt)
    return FileResponse(
        result.output_path,
        filename=result.output_path.name,
        media_type="application/octet-stream",
    )


@router.post("/knowledge/query")
async def query_knowledge(params: dict[str, Any]) -> dict[str, Any]:
    """Query the knowledge base and return relevant chunks."""
    if get_context().kb is None:
        return {"chunks": [], "error": "Knowledge base is not available"}
    try:
        text = params.get("query", "")
        top_k = int(params.get("top_k", 5))
        chunks = get_context().kb.query(text, top_k=top_k)
        return {"chunks": chunks}
    except Exception as e:
        return {"chunks": [], "error": str(e)}
