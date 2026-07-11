"""Research project (科研专题) endpoints.

Combines Claude Projects' three-part structure (instructions + KB + chat
threads) with Metaso-style topic organisation. Storage is a plain JSON file
— no database for a desktop app.
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/projects", tags=["research_projects"])

_lock = threading.Lock()
_store_path = Path.home() / ".huginn" / "research_projects.json"

# mtime-based cache: avoids reading + json.loads on every request
_cached_data: dict[str, dict[str, Any]] | None = None
_cached_mtime: float = 0.0


def _load() -> dict[str, dict[str, Any]]:
    global _cached_data, _cached_mtime
    if not _store_path.exists():
        return {}
    try:
        mtime = _store_path.stat().st_mtime
        if _cached_data is not None and mtime == _cached_mtime:
            return _cached_data
        _cached_data = json.loads(_store_path.read_text("utf-8"))
        _cached_mtime = mtime
        return _cached_data
    except (json.JSONDecodeError, OSError):
        return {}


def _invalidate_cache() -> None:
    global _cached_data
    _cached_data = None


def _save(data: dict[str, dict[str, Any]]) -> None:
    _store_path.parent.mkdir(parents=True, exist_ok=True)
    _store_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
    _invalidate_cache()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class CreateProject(BaseModel):
    title: str
    description: str = ""
    instructions: str = ""


class UpdateProject(BaseModel):
    title: str | None = None
    description: str | None = None
    instructions: str | None = None
    search_scope: str | None = None  # "local" | "web" | "both"


class AttachThread(BaseModel):
    thread_id: str


class AttachDoc(BaseModel):
    doc_id: str


@router.get("")
async def list_projects() -> dict[str, Any]:
    with _lock:
        data = _load()
    return {"projects": list(data.values()), "count": len(data)}


@router.post("")
async def create_project(req: CreateProject) -> dict[str, Any]:
    pid = str(uuid.uuid4())
    now = _now()
    project = {
        "id": pid,
        "title": req.title,
        "description": req.description,
        "instructions": req.instructions,
        "thread_ids": [],
        "knowledge_doc_ids": [],
        "created_at": now,
        "updated_at": now,
        "search_scope": "both",
    }
    with _lock:
        data = _load()
        data[pid] = project
        _save(data)
    return {"success": True, "project": project}


@router.get("/{pid}")
async def get_project(pid: str) -> dict[str, Any]:
    with _lock:
        data = _load()
    if pid not in data:
        return {"error": "not found"}
    return {"project": data[pid]}


@router.patch("/{pid}")
async def update_project(pid: str, req: UpdateProject) -> dict[str, Any]:
    with _lock:
        data = _load()
        if pid not in data:
            return {"error": "not found"}
        p = data[pid]
        for field in ("title", "description", "instructions", "search_scope"):
            val = getattr(req, field)
            if val is not None:
                p[field] = val
        p["updated_at"] = _now()
        _save(data)
    return {"success": True, "project": p}


@router.delete("/{pid}")
async def delete_project(pid: str) -> dict[str, Any]:
    with _lock:
        data = _load()
        if pid not in data:
            return {"error": "not found"}
        del data[pid]
        _save(data)
    return {"success": True}


@router.post("/{pid}/threads")
async def attach_thread(pid: str, req: AttachThread) -> dict[str, Any]:
    with _lock:
        data = _load()
        if pid not in data:
            return {"error": "not found"}
        if req.thread_id not in data[pid]["thread_ids"]:
            data[pid]["thread_ids"].append(req.thread_id)
            data[pid]["updated_at"] = _now()
            _save(data)
    return {"success": True, "project": data[pid]}


@router.delete("/{pid}/threads/{tid}")
async def detach_thread(pid: str, tid: str) -> dict[str, Any]:
    with _lock:
        data = _load()
        if pid not in data:
            return {"error": "not found"}
        data[pid]["thread_ids"] = [t for t in data[pid]["thread_ids"] if t != tid]
        data[pid]["updated_at"] = _now()
        _save(data)
    return {"success": True, "project": data[pid]}


@router.get("/{pid}/knowledge")
async def list_knowledge(pid: str) -> dict[str, Any]:
    with _lock:
        data = _load()
    if pid not in data:
        return {"error": "not found"}
    return {"doc_ids": data[pid]["knowledge_doc_ids"]}


@router.post("/{pid}/knowledge")
async def attach_knowledge(pid: str, req: AttachDoc) -> dict[str, Any]:
    with _lock:
        data = _load()
        if pid not in data:
            return {"error": "not found"}
        if req.doc_id not in data[pid]["knowledge_doc_ids"]:
            data[pid]["knowledge_doc_ids"].append(req.doc_id)
            data[pid]["updated_at"] = _now()
            _save(data)
    return {"success": True, "project": data[pid]}


@router.delete("/{pid}/knowledge/{doc_id}")
async def detach_knowledge(pid: str, doc_id: str) -> dict[str, Any]:
    with _lock:
        data = _load()
        if pid not in data:
            return {"error": "not found"}
        data[pid]["knowledge_doc_ids"] = [
            d for d in data[pid]["knowledge_doc_ids"] if d != doc_id
        ]
        data[pid]["updated_at"] = _now()
        _save(data)
    return {"success": True, "project": data[pid]}
