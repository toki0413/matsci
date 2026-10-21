"""Consolidator apply/rollback 测试：只支持可逆动作，不删除 direct 证据."""
from __future__ import annotations

from huginn.memory.dream import RepairAction, synthesize_dream_object
from huginn.memory.dream_apply import apply_repair, rollback_repair
from huginn.memory.dream_store import DreamStore


class _FakeLongTerm:
    def __init__(self) -> None:
        self.entries = {"m1": {"importance": 0.5, "archived": 0}}

    def get_by_id(self, entry_id):
        return self.entries.get(entry_id)

    def update(self, entry_id, importance=None, **kwargs):
        self.entries[entry_id]["importance"] = importance
        return True

    def update_archived(self, entry_id, archived=True):
        self.entries[entry_id]["archived"] = 1 if archived else 0
        return True


def _prepare(tmp_path, action: str, target: str = "m1"):
    store = DreamStore(tmp_path / "dream.db")
    obj = synthesize_dream_object(
        dream_id="d1",
        persona_id="p1",
        sleep_window=("a", "b"),
        findings=[],
        repairs=[],
    )
    repair = RepairAction(
        action=action, target_ids=(target,), risk="low", reversible=True
    )
    store.save(obj, [repair])
    return store, repair, "d1:0"


def test_soft_decay_apply_and_rollback(tmp_path):
    store, repair, repair_id = _prepare(tmp_path, "soft_decay")
    longterm = _FakeLongTerm()
    result = apply_repair(
        repair, repair_id=repair_id, longterm=longterm, store=store
    )
    assert result.supported and result.applied
    assert longterm.entries["m1"]["importance"] < 0.5
    record = store.get_repair(repair_id)
    assert record["applied"] == 1
    assert record["evidence_json"]["rollback"]["m1"] == 0.5

    rolled = rollback_repair(repair_id, longterm=longterm, store=store)
    assert rolled.reason == "rolled_back"
    assert longterm.entries["m1"]["importance"] == 0.5
    assert store.get_repair(repair_id)["applied"] == 0


def test_quarantine_apply_and_rollback(tmp_path):
    store, repair, repair_id = _prepare(tmp_path, "quarantine")
    longterm = _FakeLongTerm()
    result = apply_repair(
        repair, repair_id=repair_id, longterm=longterm, store=store
    )
    assert result.supported and result.applied
    assert longterm.entries["m1"]["archived"] == 1
    rollback_repair(repair_id, longterm=longterm, store=store)
    assert longterm.entries["m1"]["archived"] == 0


def test_unsupported_action_stays_proposal(tmp_path):
    store, repair, repair_id = _prepare(tmp_path, "add_link")
    longterm = _FakeLongTerm()
    result = apply_repair(
        repair, repair_id=repair_id, longterm=longterm, store=store
    )
    assert result.supported is False
    assert result.applied is False
    assert store.get_repair(repair_id)["applied"] == 0


def test_dry_run_does_not_mutate(tmp_path):
    store, repair, repair_id = _prepare(tmp_path, "soft_decay")
    longterm = _FakeLongTerm()
    result = apply_repair(
        repair, repair_id=repair_id, longterm=longterm, store=store, dry_run=True
    )
    assert result.supported is True
    assert result.applied is False
    assert longterm.entries["m1"]["importance"] == 0.5
