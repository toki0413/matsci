"""I32: 高惊奇不得绕过 Hygiene Gate，不得提升证据权威."""
from __future__ import annotations

from huginn.memory.hygiene_gate import evaluate_influence
from huginn.metacog.belief_surprise import (
    belief_surprise,
    surprise_rank,
    surrogate_surprise,
)
from huginn.research.distillation import authority_vector


def test_surprise_not_authority():
    result = belief_surprise([0.5, 0.5], [0.9, 0.1])
    assert result["method"] == "kl"
    assert result["surprise"] > 0
    proxy = surrogate_surprise(
        prediction_error=0.9, new_contradiction=True, retrieval_failure=True
    )
    assert proxy["method"] == "surrogate", "代理必须与定理版可区分"
    assert 0.0 <= proxy["surprise"] <= 1.0

    ranked = surprise_rank([{"surprise": 0.2}, {"surprise": 0.9}])
    assert ranked[0]["surprise"] == 0.9

    # 权威上限与惊奇无关：hypothesis 永远 ≤0.1 证据权威
    assert authority_vector("hypothesis").evidential_authority <= 0.1

    # 同一记忆只改 surprise，不影响 Hygiene Gate 的 route/influence
    base = {
        "memory_id": "m1",
        "importance": 1.0,
        "confidence": 1.0,
        "relevance": 1.0,
        "grounding_status": 1.0,
        "provenance_completeness": 1.0,
        "recency_validity": 1.0,
        "authority_class": "hypothesis",
    }
    low = evaluate_influence({**base, "surprise": 0.0}, query={"tenant_id": "t1"})
    high = evaluate_influence({**base, "surprise": 1.0}, query={"tenant_id": "t1"})
    assert low.route == high.route
    assert low.influence == high.influence
