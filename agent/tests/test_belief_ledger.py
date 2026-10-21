"""I6/I7: 认知账——取代保留双份、按 valid-time 检索、冲突可寻址."""
from __future__ import annotations

from huginn.memory.longterm import LongTermMemory
from huginn.research.belief_ledger import BeliefLedger, BeliefRecord


def _record(**overrides) -> BeliefRecord:
    data = {
        "belief_id": "b1",
        "tenant_id": "t1",
        "subject": "band_gap",
        "claim": "3.4 eV",
        "subject_scope": "science",
        "provenance": ("doi:10.1/a",),
        "valid_from": "2026-01-01T00:00:00+00:00",
    }
    data.update(overrides)
    return BeliefRecord(**data)


def test_supersede_keeps_both(tmp_path):
    """I7: 被取代时保留双份；取代指针可查，旧记录不删除."""
    ledger = BeliefLedger(tmp_path / "belief.db")
    old = _record()
    ledger.append(old)
    new = ledger.supersede(
        old.belief_id,
        claim="3.45 eV",
        provenance=("doi:10.1/b",),
        valid_from="2026-03-01T00:00:00+00:00",
        belief_id="b2",
    )

    assert ledger.get("b1") is not None, "旧记录必须保留"
    assert ledger.get("b2") is not None
    assert ledger.superseded_by("b1") == "b2"
    current = ledger.current("t1", "band_gap")
    assert current["belief_id"] == "b2", "检索按 valid-time 取最新"
    assert current["claim"] == "3.45 eV"
    assert len(ledger.list_records("t1", "band_gap")) == 2


def test_valid_time_ordering(tmp_path):
    ledger = BeliefLedger(tmp_path / "belief.db")
    ledger.append(_record(belief_id="b1", valid_from="2026-05-01T00:00:00+00:00"))
    ledger.append(
        _record(
            belief_id="b2", claim="3.5 eV",
            valid_from="2026-02-01T00:00:00+00:00",
        )
    )
    # 没有取代链时，current 取 valid_from 最新
    assert ledger.current("t1", "band_gap")["belief_id"] == "b1"


def test_longterm_lint_writes_superseded_by(tmp_path, monkeypatch):
    """DoD 5: 记忆矛盾升级为 superseded_by，双份保留，valid-time 可检索."""
    import huginn.research.belief_ledger as belief_mod

    monkeypatch.setenv("HUGINN_TENANT_ID", "t1")
    ledger = BeliefLedger(tmp_path / "belief.db")
    monkeypatch.setattr(belief_mod, "get_belief_ledger", lambda: ledger)

    mem = LongTermMemory(db_path=str(tmp_path / "mem.db"))
    old_id = mem.store(
        content="Fe band gap 0.0 eV", category="fact",
        tags=["band_gap"], formula="Fe", importance=0.5, tier="mid",
    )
    new_id = mem.store(
        content="Fe band gap 2.1 eV", category="fact",
        tags=["band_gap"], formula="Fe", importance=0.6, tier="mid",
    )
    result = mem.lint(auto_fix=True)
    assert result["contradictions"], result
    assert result["superseded_by"] >= 1, result

    # 双份记录都在，旧记录带 superseded_by 指针
    old_entry = mem.get_by_id(old_id)
    new_entry = mem.get_by_id(new_id)
    assert old_entry is not None and new_entry is not None
    import json

    tags = (
        json.loads(old_entry["tags"])
        if isinstance(old_entry["tags"], str)
        else old_entry["tags"]
    )
    assert any(str(t).startswith("superseded_by:") for t in tags), tags

    records = ledger.list_records("t1", "Fe:band_gap")
    assert len(records) >= 2
    current = ledger.current("t1", "Fe:band_gap")
    assert current is not None
    assert new_id in current["belief_id"]
    assert ledger.obstruction("t1", "Fe:band_gap") is not None


def test_tenant_isolation(tmp_path):
    ledger = BeliefLedger(tmp_path / "belief.db")
    ledger.append(_record(tenant_id="t1"))
    assert ledger.get("b1", tenant_id="t2") is None
    assert ledger.current("t2", "band_gap") is None
    assert ledger.list_records("t2") == []
