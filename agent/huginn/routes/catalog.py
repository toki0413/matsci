"""Catalog 清单端点 —— 统一接入项的只读视图 + 启停/卸载.

GET    /catalog         触发 discover_all 返回归一清单 (kind/name/origin/enabled).
PATCH  /catalog/{id}    启停: {"enabled": bool} 落到注册表 (MCP 断重连, 工具摘/复原).
DELETE /catalog/{id}    卸载: 从追踪视图移除 (注册表 layer 的摘除也走 enable=false 语义).
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Body, HTTPException

from huginn.catalog.reconcile import apply_enabled
from huginn.plugins.loader import DEFAULT_PLUGINS_DIR
from huginn.server_core import get_context

router = APIRouter(prefix="/catalog", tags=["catalog"])

logger = logging.getLogger(__name__)


def _catalog():
    return getattr(get_context(), "catalog", None)


@router.get("")
async def list_catalog() -> dict[str, Any]:
    """返回统一接入清单 (discover_all 后归一)."""
    cat = _catalog()
    if cat is None:
        return {"items": [], "total": 0}
    entries = cat.discover_all(
        mcp_manager=get_context().mcp_manager, plugins_dir=DEFAULT_PLUGINS_DIR
    )
    return {
        "items": [e.to_dict() for e in entries],
        "total": len(entries),
    }


@router.patch("/{entry_id}")
async def set_enabled(entry_id: str, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """启停一项: ``{"enabled": bool}``; mcp 会断/重连, 工具摘除/复原."""
    cat = _catalog()
    if cat is None:
        raise HTTPException(status_code=404, detail="catalog not available")
    enabled = payload.get("enabled")
    if not isinstance(enabled, bool):
        raise HTTPException(status_code=422, detail="'enabled' must be a bool")

    entry = cat.get(entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"unknown entry: {entry_id}")

    # 先把标记写进追踪视图, 再落地注册表.
    cat.set_enabled(entry_id, enabled)
    result = await apply_enabled(entry, enabled, get_context().mcp_manager)
    return {**cat.get(entry_id).to_dict(), "result": result}


@router.delete("/{entry_id}")
async def delete_entry(entry_id: str) -> dict[str, str]:
    """卸载一项: 先走下线语义, 再从追踪视图移除."""
    cat = _catalog()
    if cat is None:
        raise HTTPException(status_code=404, detail="catalog not available")
    entry = cat.get(entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"unknown entry: {entry_id}")

    result = await apply_enabled(entry, False, get_context().mcp_manager)
    cat.uninstall(entry_id)
    return {"ok": "true", "result": result}