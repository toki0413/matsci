"""I17: Hygiene Gate 只允许七种 route，禁止静默放行."""
from __future__ import annotations

from huginn.memory.hygiene_gate import evaluate_influence

_ALLOWED = {
    "authorize",
    "uncertain",
    "low_impact",
    "arbitrate",
    "quarantine",
    "consolidate",
    "block",
}


def test_closed_routes():
    cases = [
        {},
        {"revoked": True},
        {"expired": True},
        {"contradiction_pressure": 0.99},
        {"provenance_completeness": 0.0, "risk": "high"},
        {"confidence": 0.0},
        {"importance": 1.0, "confidence": 1.0, "relevance": 1.0,
         "grounding_status": 1.0, "provenance_completeness": 1.0,
         "recency_validity": 1.0},
        {"tenant_id": "other"},
    ]
    for memory in cases:
        decision = evaluate_influence(memory, query={"tenant_id": "tenant-a"})
        assert decision.route in _ALLOWED, decision
        if decision.route in ("block", "quarantine", "arbitrate", "consolidate"):
            assert decision.influence == 0.0, "禁止静默放行"

    contradiction = evaluate_influence(
        {"contradiction_pressure": 0.99, "importance": 1.0},
        query={"tenant_id": "tenant-a"},
    )
    assert contradiction.route == "arbitrate", "高矛盾不得静默选边"
