"""Visual perception endpoints for Huginn.

Exposes the I-JEPA-backed image encoder as HTTP endpoints so the frontend
(or any client) can:

  * encode an image into a vector          POST /visual/encode
  * add an image to the persistent index    POST /visual/index
  * search the index for similar images     POST /visual/search
  * inspect the index                       GET  /visual/index/stats

All heavy lifting lives in ``huginn.perception.visual_encoder`` and
``huginn.perception.image_index``; the route handlers stay thin and defer
blocking work to a thread pool so the event loop isn't stalled while torch
runs inference.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, File, Form, UploadFile

from huginn.server_core import get_image_index, get_visual_encoder

router = APIRouter(tags=["visual"])

# Keep a sane ceiling on uploads — SEM/TEM tiles are rarely bigger than this.
_MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB


def _parse_metadata(raw: str | None) -> dict[str, Any]:
    """Best-effort parse of an optional JSON metadata form field."""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    return {}


async def _read_capped(file: UploadFile) -> bytes:
    """Read an upload, rejecting anything over the size cap."""
    content = await file.read()
    if len(content) > _MAX_UPLOAD_BYTES:
        raise ValueError(
            f"image too large ({len(content)} bytes, max {_MAX_UPLOAD_BYTES})"
        )
    return content


@router.post("/visual/encode")
async def visual_encode(
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """Encode an uploaded image and return its embedding vector."""
    try:
        content = await _read_capped(file)
    except ValueError as exc:
        return {"success": False, "error": str(exc)}

    encoder = get_visual_encoder()
    if encoder is None or not encoder.available:
        return {
            "success": False,
            "error": "no visual encoder backend available",
            "init_error": encoder.init_error if encoder else "encoder not initialized",
        }

    # torch inference is blocking — push it off the event loop.
    import asyncio

    vec = await asyncio.to_thread(encoder.encode_image, content)
    if vec is None:
        return {"success": False, "error": "encoding failed for this image"}

    return {
        "success": True,
        "backend": encoder.backend_name,
        "dim": encoder.dim,
        "vector": vec.tolist(),
        "filename": file.filename,
    }


@router.post("/visual/index")
async def visual_index(
    file: UploadFile = File(...),
    metadata: str = Form(""),
) -> dict[str, Any]:
    """Add an uploaded image to the shared image index."""
    try:
        content = await _read_capped(file)
    except ValueError as exc:
        return {"success": False, "error": str(exc)}

    index = get_image_index()
    meta = _parse_metadata(metadata)
    meta.setdefault("filename", file.filename or "")

    import asyncio

    record = await asyncio.to_thread(index.add_image_bytes, content, meta, file.filename)

    # Persist so the index survives restarts.
    try:
        await asyncio.to_thread(index.save)
    except Exception as exc:  # noqa: BLE001 - persistence shouldn't fail the request
        return {
            "success": True,
            "record": {k: v for k, v in record.items() if k != "vector"},
            "warning": f"index not persisted: {exc}",
        }

    # The raw vector is bulky and not interesting to the client; drop it.
    public = {k: v for k, v in record.items() if k != "vector"}
    return {"success": True, "record": public}


@router.post("/visual/search")
async def visual_search(
    file: UploadFile = File(...),
    top_k: int = Form(5),
) -> dict[str, Any]:
    """Search the image index for entries similar to the uploaded image."""
    try:
        content = await _read_capped(file)
    except ValueError as exc:
        return {"success": False, "error": str(exc)}

    index = get_image_index()
    encoder = get_visual_encoder()
    if encoder is None or not encoder.available:
        return {
            "success": False,
            "results": [],
            "error": "no visual encoder backend available",
        }

    if len(index) == 0:
        return {"success": True, "results": [], "message": "index is empty"}

    import asyncio

    results = await asyncio.to_thread(index.search, content, top_k)
    return {"success": True, "results": results, "count": len(results)}


@router.get("/visual/index/stats")
async def visual_index_stats() -> dict[str, Any]:
    """Return summary statistics for the shared image index."""
    index = get_image_index()
    encoder = get_visual_encoder()
    stats = index.stats()
    # Surface encoder diagnostics too, so the client can tell *why*
    # embedding_dim might be 0.
    stats["encoder_error"] = encoder.init_error if encoder else None
    return stats
