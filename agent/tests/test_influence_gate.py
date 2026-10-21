"""I14/I24: 影响权威逐查询 + 双重 clamp；注入必须带来源标签."""
from __future__ import annotations

from huginn.memory.hygiene_gate import assemble_context, evaluate_influence


def _strong(**overrides) -> dict:
    data = {
        "memory_id": "m1",
        "importance": 1.0,
        "confidence": 1.0,
        "relevance": 1.0,
        "grounding_status": 1.0,
        "provenance_completeness": 1.0,
        "recency_validity": 1.0,
        "uncertainty": 0.0,
        "contradiction_pressure": 0.0,
        "text": "evidence",
    }
    data.update(overrides)
    return data


def test_per_query_cap():
    """I14: influence 被 authority_class 与 route 双重 clamp，逐查询计算."""
    direct = evaluate_influence(_strong(authority_class="direct"))
    assert direct.route == "authorize"
    assert direct.influence == 1.0

    derived = evaluate_influence(_strong(authority_class="derived"))
    assert derived.influence <= 0.6

    hypothesis = evaluate_influence(_strong(authority_class="hypothesis"))
    assert hypothesis.influence <= 0.4

    # 同一记忆换 query：跨租户 → 0（逐查询，不是全局缓存）
    capped = evaluate_influence(
        _strong(authority_class="direct", tenant_id="tenant-b"),
        query={"tenant_id": "tenant-a"},
    )
    assert capped.route == "block" and capped.influence == 0.0


def test_labeled_injection():
    """I24: 每条注入记忆必须带来源标签；全 block 时显式声明无证据."""
    pack = assemble_context(
        [
            _strong(memory_id="d", authority_class="direct"),
            _strong(memory_id="v", authority_class="derived"),
            _strong(memory_id="h", authority_class="hypothesis"),
        ],
        query={"tenant_id": "tenant-a"},
        budget=3.0,
    )
    assert not pack.no_evidence
    labels = {entry.memory_id: entry.label for entry in pack.entries}
    assert labels["d"].startswith("[DIRECT]")
    assert labels["v"].startswith("[DERIVED]")
    assert labels["h"].startswith("[DREAM]")
    for entry in pack.entries:
        assert entry.label.startswith("["), "禁止无标签注入"

    empty = assemble_context(
        [_strong(memory_id="x", revoked=True)], query={"tenant_id": "tenant-a"}
    )
    assert empty.no_evidence is True
    assert not empty.entries
