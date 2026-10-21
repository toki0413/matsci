"""记忆双层治理测试：保留/影响独立、权威上限、封闭路由、无证据诚实."""
from __future__ import annotations

from huginn.memory.hygiene_gate import (
    assemble_context,
    evaluate_influence,
    evaluate_retention,
)


def test_retention_and_influence_are_independent():
    """I13: 保留可用不代表可进入上下文；影响 block/arbitrate 不代表删除."""
    memory = {
        "memory_id": "m1",
        "importance": 0.9,
        "contradiction_pressure": 0.95,
        "authority_class": "direct",
    }
    retention = evaluate_retention(memory)
    influence = evaluate_influence(memory, query={"tenant_id": "t1"})
    assert retention.action == "keep_available"
    assert influence.route == "arbitrate"
    assert influence.influence == 0.0
    assert "仲裁" in influence.reasons[0]


def test_authority_class_caps_influence():
    """derived ≤ 0.6 / hypothesis ≤ 0.4，即使评分很高."""
    query = {"query_id": "q1", "tenant_id": "t1"}
    derived = evaluate_influence(
        {
            "memory_id": "d1",
            "authority_class": "derived",
            "relevance": 1.0,
            "confidence": 1.0,
            "grounding_status": 1.0,
            "provenance_completeness": 1.0,
            "recency_validity": 1.0,
        },
        query=query,
    )
    hypothesis = evaluate_influence(
        {
            "memory_id": "h1",
            "authority_class": "hypothesis",
            "relevance": 1.0,
            "confidence": 1.0,
            "grounding_status": 1.0,
            "provenance_completeness": 1.0,
            "recency_validity": 1.0,
        },
        query=query,
    )
    assert derived.influence <= 0.6
    assert hypothesis.influence <= 0.4
    assert hypothesis.influence < derived.influence


def test_high_contradiction_routes_to_arbitrate():
    decision = evaluate_influence(
        {
            "memory_id": "m1",
            "contradiction_pressure": 0.95,
            "confidence": 0.9,
            "relevance": 0.9,
        }
    )
    assert decision.route == "arbitrate"
    assert decision.influence == 0.0


def test_expired_and_cross_tenant_block():
    expired = evaluate_influence({"memory_id": "m1", "expired": True})
    assert expired.route == "block"
    cross = evaluate_influence(
        {"memory_id": "m2", "tenant_id": "tenant-b"},
        query={"tenant_id": "tenant-a"},
    )
    assert cross.route == "block"
    assert "跨租户" in cross.reasons[0]


def test_assemble_context_orders_by_influence_and_labels():
    pack = assemble_context(
        [
            {
                "memory_id": "low",
                "text": "low",
                "authority_class": "hypothesis",
                "relevance": 0.2,
                "confidence": 0.4,
                "provenance_completeness": 0.2,
            },
            {
                "memory_id": "high",
                "text": "high",
                "authority_class": "direct",
                "relevance": 1.0,
                "confidence": 1.0,
                "grounding_status": 1.0,
                "provenance_completeness": 1.0,
            },
        ],
        query={"query_id": "q1", "tenant_id": "t1"},
        budget=2.0,
    )
    assert not pack.no_evidence
    assert pack.entries[0].memory_id == "high"
    assert pack.entries[0].label.startswith("[DIRECT]")
    assert any(e.label.startswith("[DREAM]") or "[LOW]" in e.label for e in pack.entries)


def test_all_blocked_is_explicit_no_evidence():
    pack = assemble_context(
        [
            {"memory_id": "m1", "revoked": True},
            {"memory_id": "m2", "expired": True},
        ],
        query={"query_id": "q1", "tenant_id": "t1"},
    )
    assert pack.no_evidence is True
    assert not pack.entries


def test_prune_requires_two_signals():
    """I20: 过期默认只归档；硬修剪需要第二个独立信号 + allow_prune."""
    memory = {"memory_id": "m1", "expired": True}
    assert evaluate_retention(memory).action == "archive"
    assert evaluate_retention(memory, allow_prune=True).action == "archive"
    memory["predicted_failure"] = True
    assert evaluate_retention(memory, allow_prune=True).action == "prune"


def test_dream_candidate_cannot_authorize_action():
    """I15/I32: 梦境/假设产物即使高分也只能低影响，不能获得 direct 权威."""
    decision = evaluate_influence(
        {
            "memory_id": "dream-1",
            "authority_class": "hypothesis",
            "relevance": 1.0,
            "confidence": 1.0,
            "grounding_status": 1.0,
            "provenance_completeness": 1.0,
        }
    )
    assert decision.authority_class == "hypothesis"
    assert decision.influence <= 0.4
    assert decision.route in ("authorize", "uncertain", "low_impact")
