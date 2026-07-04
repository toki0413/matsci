"""Unified scientific computing framework endpoints."""

from __future__ import annotations

import base64
import logging
import traceback
from typing import Any

from fastapi import APIRouter

router = APIRouter(tags=["unified"])

logger = logging.getLogger(__name__)


@router.get("/unified/models")
async def unified_models() -> dict[str, Any]:
    """List unified models and multiscale bridges."""
    from huginn.unified.bridge import list_bridges
    from huginn.unified.models import list_models

    return {"models": list_models(), "bridges": list_bridges()}


@router.post("/unified/derive")
async def unified_derive(params: dict[str, Any]) -> dict[str, Any]:
    """Derive governing equations for a unified model."""
    from huginn.unified import derive_equations
    from huginn.unified.models import get_model

    model_name = params.get("model")
    factory = get_model(model_name)
    if not factory:
        return {"success": False, "error": f"Model '{model_name}' not found"}
    try:
        problem = factory()
        result = derive_equations(problem)
        return {
            "success": True,
            "problem": problem.to_dict(),
            "principle": result["principle"],
            "equations": {k: str(v) for k, v in result["equations"].items()},
        }
    except Exception as e:
        logger.error("unexpected error", exc_info=True)
        return {"success": False, "error": str(e)}


@router.post("/unified/solve")
async def unified_solve_endpoint(params: dict[str, Any]) -> dict[str, Any]:
    """Discretize and solve a unified model."""
    from huginn.unified import solve
    from huginn.unified.models import get_model

    model_name = params.get("model")
    factory = get_model(model_name)
    if not factory:
        return {"success": False, "error": f"Model '{model_name}' not found"}
    try:
        problem = factory()
        result = solve(
            problem, method=params.get("method", "fem"), n=params.get("n", 10)
        )
        return {"success": True, **result}
    except Exception as e:
        logger.error("unexpected error", exc_info=True)
        return {"success": False, "error": str(e)}


@router.post("/unified/plot")
async def unified_plot_endpoint(params: dict[str, Any]) -> dict[str, Any]:
    """Solve a unified model and return a plot."""
    from huginn.unified import solve_and_plot
    from huginn.unified.models import get_model

    model_name = params.get("model")
    factory = get_model(model_name)
    if not factory:
        return {"success": False, "error": f"Model '{model_name}' not found"}
    try:
        problem = factory()
        output_path = params.get("output_path", "unified_solution.png")
        result = solve_and_plot(
            problem,
            method=params.get("method", "fem"),
            n=params.get("n", 10),
            output_path=output_path,
        )
        with open(result["plot_path"], "rb") as fimg:
            img_bytes = fimg.read()
        return {
            "success": True,
            "plot_path": result["plot_path"],
            "plot_base64": base64.b64encode(img_bytes).decode("utf-8"),
            "residual": result["residual"],
            "n_dof": result["n_dof"],
        }
    except Exception as e:
        logger.error("unexpected error", exc_info=True)
        return {"success": False, "error": str(e)}
