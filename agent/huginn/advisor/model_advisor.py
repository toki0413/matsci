"""Model advisor — recommends LLM models based on user needs."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from huginn.advisor.knowledge import ModelProfile, get_profiles, find_by_task

@dataclass
class Recommendation:
    model: str
    provider: str
    reason: str
    score: float  # 0-1, higher is better
    cost_estimate: str = ""

class ModelAdvisor:
    """Recommends LLM models based on task requirements and constraints."""

    def recommend(
        self,
        task: str | None = None,
        budget: str | None = None,  # "low", "medium", "high"
        privacy: bool = False,
        context_size: str | None = None,  # "small", "medium", "large"
    ) -> list[Recommendation]:
        """Generate model recommendations based on criteria."""
        profiles = get_profiles()
        scored: list[tuple[float, ModelProfile]] = []

        for p in profiles:
            score = 0.5  # base score

            # Task match
            if task:
                task_lower = task.lower()
                for tag in p.best_for:
                    if tag in task_lower or task_lower in tag:
                        score += 0.2

            # Budget
            if budget == "low" and p.cost_per_1k_tokens < 0.001:
                score += 0.2
            elif budget == "low" and p.cost_per_1k_tokens >= 0.005:
                score -= 0.2
            elif budget == "high" and p.cost_per_1k_tokens >= 0.003:
                score += 0.1  # premium models often better

            # Privacy
            if privacy and not p.requires_api_key:
                score += 0.3
            elif privacy and p.requires_api_key:
                score -= 0.1

            # Context size
            if context_size == "large" and p.context_window >= 100000:
                score += 0.2
            elif context_size == "large" and p.context_window < 32000:
                score -= 0.2

            scored.append((score, p))

        scored.sort(key=lambda x: x[0], reverse=True)

        recommendations = []
        for score, profile in scored[:5]:
            reason_parts = []
            if task:
                matching = [t for t in profile.best_for if task.lower() in t or t in task.lower()]
                if matching:
                    reason_parts.append(f"Good for {task}")
            if privacy and not profile.requires_api_key:
                reason_parts.append("Local/private deployment")
            if budget == "low" and profile.cost_per_1k_tokens < 0.001:
                reason_parts.append("Cost-effective")
            if not reason_parts:
                reason_parts.extend(profile.strengths[:2])

            recommendations.append(Recommendation(
                model=profile.name,
                provider=profile.provider,
                reason=". ".join(reason_parts),
                score=round(score, 2),
                cost_estimate=f"${profile.cost_per_1k_tokens:.4f}/1K tokens" if profile.cost_per_1k_tokens > 0 else "Free (local)",
            ))

        return recommendations

    def compare(self, model_a: str, model_b: str) -> dict[str, Any]:
        """Compare two models side-by-side."""
        profiles = get_profiles()
        a = next((p for p in profiles if p.name.lower() == model_a.lower()), None)
        b = next((p for p in profiles if p.name.lower() == model_b.lower()), None)

        if not a or not b:
            return {"error": f"Model not found: {model_a if not a else model_b}"}

        return {
            "model_a": {"name": a.name, "provider": a.provider, "context_window": a.context_window,
                        "cost": a.cost_per_1k_tokens, "strengths": a.strengths, "weaknesses": a.weaknesses},
            "model_b": {"name": b.name, "provider": b.provider, "context_window": b.context_window,
                        "cost": b.cost_per_1k_tokens, "strengths": b.strengths, "weaknesses": b.weaknesses},
        }
