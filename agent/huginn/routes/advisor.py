"""Model advisor endpoints — recommend and compare LLM models."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from huginn.advisor.knowledge import get_profiles, find_by_task, find_by_provider
from huginn.advisor.model_advisor import ModelAdvisor

router = APIRouter(tags=["advisor"])

_advisor = ModelAdvisor()


@router.get("/advisor/models")
async def list_models() -> dict[str, Any]:
    """List all known LLM model profiles."""
    profiles = get_profiles()
    return {
        "models": [
            {
                "name": p.name,
                "provider": p.provider,
                "strengths": p.strengths,
                "weaknesses": p.weaknesses,
                "context_window": p.context_window,
                "cost_per_1k_tokens": p.cost_per_1k_tokens,
                "best_for": p.best_for,
                "available": p.available,
                "requires_api_key": p.requires_api_key,
                "notes": p.notes,
            }
            for p in profiles
        ],
        "count": len(profiles),
    }


@router.post("/advisor/recommend")
async def recommend_models(params: dict[str, Any]) -> dict[str, Any]:
    """Get model recommendations based on task, budget, privacy, and context size."""
    task = params.get("task")
    budget = params.get("budget")
    privacy = params.get("privacy", False)
    context_size = params.get("context_size")

    recommendations = _advisor.recommend(
        task=task,
        budget=budget,
        privacy=privacy,
        context_size=context_size,
    )
    return {
        "recommendations": [
            {
                "model": r.model,
                "provider": r.provider,
                "reason": r.reason,
                "score": r.score,
                "cost_estimate": r.cost_estimate,
            }
            for r in recommendations
        ],
    }


@router.get("/advisor/compare")
async def compare_models(model_a: str, model_b: str) -> dict[str, Any]:
    """Compare two models side-by-side."""
    result = _advisor.compare(model_a, model_b)
    return result
