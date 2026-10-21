"""I13: 保留闸与影响闸独立——保留不提升权威，影响不删除记忆."""
from __future__ import annotations

from huginn.memory.hygiene_gate import evaluate_influence, evaluate_retention


def test_independence():
    # 保留可用，但跨租户 → 影响闸 block：两闸结论可以相反
    memory = {
        "memory_id": "m1",
        "importance": 0.9,
        "tenant_id": "tenant-b",
        "text": "keep me",
    }
    retention = evaluate_retention(memory)
    influence = evaluate_influence(memory, query={"tenant_id": "tenant-a"})
    assert retention.action == "keep_available"
    assert influence.route == "block"
    assert influence.influence == 0.0

    # 影响闸只判定、不删除：memory 字典不被修改
    before = dict(memory)
    evaluate_influence(memory, query={"tenant_id": "tenant-a"})
    assert memory == before

    # 保留为 archive 的记忆仍可被影响闸低权威召回复核
    archived = {"memory_id": "m2", "archived": 1, "importance": 0.9}
    assert evaluate_retention(archived).action == "archive"
    reviewed = evaluate_influence(archived, query={"tenant_id": "tenant-a"})
    assert reviewed.route in ("authorize", "uncertain", "low_impact")
    assert reviewed.influence > 0.0
