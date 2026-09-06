"""Admin endpoints for runtime operations."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from huginn.middleware.maintenance import is_maintenance_mode, set_maintenance_mode
from huginn.security.auth import require_admin_key

router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(require_admin_key)],
)


@router.get("/maintenance")
async def maintenance_status() -> dict[str, Any]:
    """Check if maintenance mode is active."""
    return {"maintenance": is_maintenance_mode()}


@router.post("/maintenance")
async def toggle_maintenance(params: dict[str, Any]) -> dict[str, Any]:
    """Enable or disable maintenance mode.

    Body: ``{"enabled": true|false}``
    """
    enabled = bool(params.get("enabled", False))
    set_maintenance_mode(enabled)
    return {"maintenance": is_maintenance_mode()}
