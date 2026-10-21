"""DreamStore 测试：梦对象与修复提案落库、幂等、显式 applied."""
from __future__ import annotations

from huginn.memory.dream import RepairAction, synthesize_dream_object
from huginn.memory.dream_store import DreamStore


def _store(tmp_path) -> DreamStore:
    return DreamStore(tmp_path / "dream.db")


def test_save_and_get_dream(tmp_path):
    store = _store(tmp_path)
    obj = synthesize_dream_object(
        dream_id="d1",
        persona_id="p1",
        sleep_window=("a", "b"),
        findings=[],
        repairs=[],
    )
    store.save(obj, [])
    got = store.get("d1")
    assert got is not None
    assert got["persona_id"] == "p1"
    assert got["payload"]["status"] == "candidate"
    assert store.list_dreams(persona_id="p1")


def test_repairs_are_proposals_until_explicitly_applied(tmp_path):
    store = _store(tmp_path)
    obj = synthesize_dream_object(
        dream_id="d2",
        persona_id="p1",
        sleep_window=("a", "b"),
        findings=[],
        repairs=[],
    )
    repair = RepairAction(
        action="soft_decay",
        target_ids=("m1",),
        risk="low",
        reversible=True,
    )
    store.save(obj, [repair])
    repairs = store.list_repairs("d2")
    assert len(repairs) == 1
    assert repairs[0]["applied"] == 0
    assert repairs[0]["action"] == "soft_decay"
    assert store.mark_repair_applied(repairs[0]["repair_id"]) is True
    assert store.list_repairs("d2")[0]["applied"] == 1


def test_dream_tenant_isolation(tmp_path):
    """§13/I22: DreamObject 带 tenant/deployment，跨租户不可见."""
    store = _store(tmp_path)
    t1 = synthesize_dream_object(
        dream_id="d-t1", persona_id="p1", sleep_window=("a", "b"),
        findings=[], repairs=[], tenant_id="t1", deployment_id="dpl-a",
    )
    t2 = synthesize_dream_object(
        dream_id="d-t2", persona_id="p1", sleep_window=("a", "b"),
        findings=[], repairs=[], tenant_id="t2", deployment_id="dpl-b",
    )
    store.save(t1, [])
    store.save(t2, [])

    only_t1 = store.list_dreams(tenant_id="t1")
    assert [item["dream_id"] for item in only_t1] == ["d-t1"]
    assert store.get("d-t1")["tenant_id"] == "t1"
    assert store.get("d-t1")["deployment_id"] == "dpl-a"
    assert store.get("d-t2")["tenant_id"] == "t2"
    # 不传 tenant 时保留旧行为（跨租户管理视图）
    assert len(store.list_dreams(persona_id="p1")) == 2


def test_save_is_idempotent(tmp_path):
    store = _store(tmp_path)
    obj = synthesize_dream_object(
        dream_id="d3",
        persona_id="p1",
        sleep_window=("a", "b"),
        findings=[],
        repairs=[],
    )
    store.save(obj, [])
    store.save(obj, [])
    assert len(store.list_dreams(persona_id="p1")) == 1
