"""Tunable parameters routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/parameters", tags=["parameters"])


class ParamSetRequest(BaseModel):
    name: str
    value: Any


class ParamResponse(BaseModel):
    name: str
    type: str
    current: Any
    default: Any
    description: str
    category: str


@router.get("/", response_model=list[ParamResponse])
async def list_parameters(category: str | None = None) -> list[dict[str, Any]]:
    """List all tunable parameters, optionally filtered by category."""
    from huginn.tools.parameters import ParameterRegistry

    return ParameterRegistry.list_params(category=category)


@router.get("/categories")
async def list_categories() -> dict[str, list[str]]:
    """Return all available parameter categories."""
    from huginn.tools.parameters import ParameterRegistry

    return {"categories": ParameterRegistry.categories()}


@router.post("/set")
async def set_parameter(request: ParamSetRequest) -> dict[str, Any]:
    """Set a parameter value at runtime."""
    from huginn.tools.parameters import ParameterRegistry

    ok, msg = ParameterRegistry.set_value(request.name, request.value)
    if not ok:
        return {"success": False, "error": msg}
    param = ParameterRegistry.get(request.name)
    return {"success": True, "name": request.name, "value": param.current}


@router.post("/reset")
async def reset_parameters(name: str | None = None) -> dict[str, Any]:
    """Reset one or all parameters to their default values."""
    from huginn.tools.parameters import ParameterRegistry

    ParameterRegistry.reset(name)
    return {"success": True, "reset": name or "all"}


@router.get("/{name}")
async def get_parameter(name: str) -> dict[str, Any]:
    """Get a single parameter by name."""
    from huginn.tools.parameters import ParameterRegistry

    param = ParameterRegistry.get(name)
    if not param:
        return {"success": False, "error": f"Unknown parameter: {name}"}
    return {
        "name": param.name,
        "type": param.param_type.value,
        "current": param.current,
        "default": param.default,
        "description": param.description,
        "category": param.category,
    }
