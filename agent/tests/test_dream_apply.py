"""Consolidator apply/rollback 测试：七类可逆修复，不删除 direct 证据."""
from __future__ import annotations

from huginn.memory.dream import (
    RepairAction,
    diagnose,
    probe_memory_graph,
    synthesize_dream_object,
)
from huginn.memory.dream_apply import apply_repair, rollback_repair
from huginn.memory.dream_store import DreamStore


class _FakeLongTerm:
    def __init__(self) -> None:
        self.entries: dict[str, dict] = {
            "m1": {
                "importance": 0.5, "archived": 0, "tags": [],
                "tier": "mid", "content": "c1",
            },
            "m2": {
                "importance": 0.6, "archived": 0, "tags": [],
                "tier": "mid", "content": "c2",
            },
            "dup": {
                "importance": 0.4, "archived": 0, "tags": ["keep"],
                "tier": "mid", "content": "dup",
            },
            "orig": {
                "importance": 0.7, "archived": 0, "tags": [],
                "tier": "mid", "content": "orig",
                "source": "run:42",
            },
        }
        self.stored: dict[str, dict] = {}

    def get_by_id(self, entry_id):
        return self.entries.get(entry_id)

    def update(
        self, entry_id, importance=None, tags=None, tier=None, content=None, **kwargs
    ):
        entry = self.entries.get(entry_id)
        if entry is None:
            return False
        if importance is not None:
            entry["importance"] = importance
        if tags is not None:
            entry["tags"] = list(tags)
        if tier is not None:
            entry["tier"] = tier
        if content is not None:
            entry["content"] = content
        return True

    def update_archived(self, entry_id, archived=True):
        entry = self.entries.get(entry_id)
        if entry is None:
            return False
        entry["archived"] = 1 if archived else 0
        return True

    def store(self, content, **kwargs):
        summary_id = f"summary_{len(self.stored) + 1}"
        self.stored[summary_id] = {"content": content, **kwargs}
        self.entries[summary_id] = {
            "importance": kwargs.get("importance", 0.5),
            "archived": 0,
            "tags": list(kwargs.get("tags") or []),
            "tier": kwargs.get("tier", "mid"),
            "content": content,
        }
        return summary_id


def _prepare(tmp_path, action: str, targets=("m1",), evidence=None):
    store = DreamStore(tmp_path / "dream.db")
    obj = synthesize_dream_object(
        dream_id="d1",
        persona_id="p1",
        sleep_window=("a", "b"),
        findings=[],
        repairs=[],
    )
    repair = RepairAction(
        action=action,
        target_ids=tuple(targets),
        risk="low",
        reversible=True,
        evidence=evidence or {},
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


def test_unknown_action_stays_proposal(tmp_path):
    store, repair, repair_id = _prepare(tmp_path, "delete")
    longterm = _FakeLongTerm()
    result = apply_repair(
        repair, repair_id=repair_id, longterm=longterm, store=store
    )
    assert result.supported is False
    assert result.applied is False
    assert store.get_repair(repair_id)["applied"] == 0


def test_merge_duplicate_keeps_both_and_rolls_back(tmp_path):
    store, repair, repair_id = _prepare(
        tmp_path, "merge_duplicate", targets=("dup", "orig")
    )
    longterm = _FakeLongTerm()
    result = apply_repair(
        repair, repair_id=repair_id, longterm=longterm, store=store
    )
    assert result.supported and result.applied
    assert longterm.entries["dup"]["archived"] == 1
    assert "duplicate_of:orig" in longterm.entries["dup"]["tags"]
    assert "keep" in longterm.entries["dup"]["tags"]
    # 原件不动，两份都保留
    assert longterm.entries["orig"]["archived"] == 0

    rollback_repair(repair_id, longterm=longterm, store=store)
    assert longterm.entries["dup"]["archived"] == 0
    assert longterm.entries["dup"]["tags"] == ["keep"]


def test_add_link_requires_target_else_proposal(tmp_path):
    store, repair, repair_id = _prepare(tmp_path, "add_link", targets=("m1",))
    longterm = _FakeLongTerm()
    result = apply_repair(
        repair, repair_id=repair_id, longterm=longterm, store=store
    )
    assert result.supported is True
    assert result.applied is False
    assert "link_to" in result.reason
    assert store.get_repair(repair_id)["applied"] == 0


def test_add_link_apply_and_rollback(tmp_path):
    store, repair, repair_id = _prepare(
        tmp_path, "add_link", targets=("m1",), evidence={"link_to": "m2"}
    )
    longterm = _FakeLongTerm()
    result = apply_repair(
        repair, repair_id=repair_id, longterm=longterm, store=store
    )
    assert result.supported and result.applied
    assert "link:m2" in longterm.entries["m1"]["tags"]
    rollback_repair(repair_id, longterm=longterm, store=store)
    assert longterm.entries["m1"]["tags"] == []


def test_summarize_writes_summary_and_archives_sources(tmp_path):
    store, repair, repair_id = _prepare(
        tmp_path,
        "summarize",
        targets=("m1", "m2"),
        evidence={"summary": "merged summary", "source": "run:9"},
    )
    longterm = _FakeLongTerm()
    result = apply_repair(
        repair, repair_id=repair_id, longterm=longterm, store=store
    )
    assert result.supported and result.applied
    summary_id = result.handle["summary_id"]
    assert longterm.stored[summary_id]["content"] == "merged summary"
    assert "cluster_summary" in longterm.stored[summary_id]["tags"]
    assert longterm.entries["m1"]["archived"] == 1
    assert longterm.entries["m2"]["archived"] == 1

    rollback_repair(repair_id, longterm=longterm, store=store)
    assert longterm.entries["m1"]["archived"] == 0
    assert longterm.entries["m2"]["archived"] == 0
    assert longterm.entries[summary_id]["archived"] == 1  # 摘要归档而非删除


def test_summarize_without_text_stays_proposal(tmp_path):
    store, repair, repair_id = _prepare(tmp_path, "summarize", targets=("m1",))
    longterm = _FakeLongTerm()
    result = apply_repair(
        repair, repair_id=repair_id, longterm=longterm, store=store
    )
    assert result.supported is True and result.applied is False
    assert "evidence.summary" in result.reason


def test_split_cluster_removes_tags_and_rolls_back(tmp_path):
    store, repair, repair_id = _prepare(tmp_path, "split_cluster", targets=("m1",))
    longterm = _FakeLongTerm()
    longterm.entries["m1"]["tags"] = ["cluster_summary", "cluster_summary_2026", "a"]
    longterm.entries["m1"]["archived"] = 1
    result = apply_repair(
        repair, repair_id=repair_id, longterm=longterm, store=store
    )
    assert result.supported and result.applied
    assert longterm.entries["m1"]["tags"] == ["a"]
    assert longterm.entries["m1"]["archived"] == 0

    rollback_repair(repair_id, longterm=longterm, store=store)
    assert longterm.entries["m1"]["tags"] == [
        "cluster_summary", "cluster_summary_2026", "a"
    ]
    assert longterm.entries["m1"]["archived"] == 1


def test_re_anchor_requires_verifiable_anchor(tmp_path):
    store, repair, repair_id = _prepare(tmp_path, "re_anchor", targets=("m1",))
    longterm = _FakeLongTerm()
    result = apply_repair(
        repair, repair_id=repair_id, longterm=longterm, store=store
    )
    assert result.supported is True and result.applied is False
    assert "anchor" in result.reason


def test_re_anchor_extends_retention_and_rolls_back(tmp_path):
    store, repair, repair_id = _prepare(tmp_path, "re_anchor", targets=("orig",))
    longterm = _FakeLongTerm()
    result = apply_repair(
        repair, repair_id=repair_id, longterm=longterm, store=store
    )
    assert result.supported and result.applied
    assert longterm.entries["orig"]["tier"] == "long"
    rollback_repair(repair_id, longterm=longterm, store=store)
    assert longterm.entries["orig"]["tier"] == "mid"


def test_dry_run_does_not_mutate(tmp_path):
    store, repair, repair_id = _prepare(tmp_path, "soft_decay")
    longterm = _FakeLongTerm()
    result = apply_repair(
        repair, repair_id=repair_id, longterm=longterm, store=store, dry_run=True
    )
    assert result.supported is True
    assert result.applied is False
    assert longterm.entries["m1"]["importance"] == 0.5


def test_duplicate_diagnosis_targets_both_ids():
    finding = probe_memory_graph([{"memory_id": "dup", "duplicate_of": "orig"}])[0]
    dx = diagnose(finding)
    assert dx.affected_ids == ("dup", "orig")
