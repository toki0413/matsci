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
