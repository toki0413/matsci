"""记忆治理账测试：追加式、租户隔离、召回影响判定留痕."""
from __future__ import annotations

from pathlib import Path

from huginn.memory.governance import MemoryStateEvent, MemoryStateLedger
from huginn.memory.hygiene_gate import InfluenceDecision


def _ledger(tmp_path) -> MemoryStateLedger:
    return MemoryStateLedger(tmp_path / "memory_state.db")


def test_audit_completeness(tmp_path):
    """I16: 所有记忆迁移类型都产生事件，且带 policy_id/version."""
    from huginn.memory.governance import MemoryStateEvent, MemoryStateLedger

    ledger = MemoryStateLedger(tmp_path / "mem.db")
    event_types = (
        "ingest", "index", "consolidate", "recall", "revise",
        "forget", "retrain", "promote", "quarantine",
    )
    for event_type in event_types:
        ledger.record(
            MemoryStateEvent(
                tenant_id="t1",
                memory_id="m1",
                event_type=event_type,
                policy_id="policy",
                policy_version="v1",
                decision=event_type,
                authority={"abstraction": 0.5, "evidential": 0.5, "influence": 0.5},
                evidence={"source": "test"},
                actor="system",
                soul_scope="agent",
            )
        )
    rows = ledger.list_events(tenant_id="t1", memory_id="m1", limit=100)
    assert {row["event_type"] for row in rows} == set(event_types)
    for row in rows:
        assert row["policy_id"] == "policy"
        assert row["policy_version"] == "v1"
        assert row["decision"]
        assert row["evidence_json"]


def test_record_and_list_events(tmp_path):
    ledger = _ledger(tmp_path)
    decision = InfluenceDecision(
        memory_id="m1",
        route="authorize",
        influence=0.8,
        authority_class="direct",
    )
    ledger.record_influence(tenant_id="t1", memory_id="m1", decision=decision)
    ledger.record_retention(
        tenant_id="t1",
        memory_id="m1",
        decision=type("R", (), {"action": "keep_available"})(),
    )
    events = ledger.list_events(tenant_id="t1", memory_id="m1")
    assert len(events) == 2
    assert events[0]["event_type"] == "recall"
    assert events[0]["decision"] == "authorize"
    assert events[0]["authority_json"]["influence"] == 0.8
    assert events[1]["event_type"] == "forget"


def test_tenant_isolation(tmp_path):
    ledger = _ledger(tmp_path)
    ledger.record(
        MemoryStateEvent(
            tenant_id="t1",
            memory_id="m1",
            event_type="ingest",
            policy_id="p",
            policy_version="v1",
            decision="keep_available",
        )
    )
    assert ledger.list_events(tenant_id="t2") == []


def test_event_id_is_idempotent(tmp_path):
    ledger = _ledger(tmp_path)
    event = MemoryStateEvent(
        tenant_id="t1",
        memory_id="m1",
        event_type="ingest",
        policy_id="p",
        policy_version="v1",
        decision="keep_available",
        event_id="fixed-id",
    )
    ledger.record(event)
    ledger.record(event)
    assert len(ledger.list_events(tenant_id="t1")) == 1


def test_recall_for_prompt_emits_influence_events(tmp_path, monkeypatch):
    """集成: recall_for_prompt 过影响闸后写审计事件."""
    from huginn.memory import governance
    from huginn.memory.longterm import LongTermMemory
    from huginn.memory.manager import MemoryManager

    ledger = _ledger(tmp_path)
    monkeypatch.setattr(governance, "_ledger", ledger)
    monkeypatch.setenv("HUGINN_MEMORY_AUDIT", "1")
    monkeypatch.setenv("HUGINN_MEMORY_INFLUENCE_GATE", "1")
    monkeypatch.setenv("HUGINN_TENANT_ID", "tenant-test")

    mem = LongTermMemory(db_path=Path(tmp_path) / "memory.db")
    mgr = MemoryManager(longterm=mem)
    mgr.remember("Si band gap is 1.1 eV", category="fact", tags=["Si", "band_gap"])
    prompt = mgr.recall_for_prompt("band gap")
    assert "Si band gap" in prompt

    events = ledger.list_events(tenant_id="tenant-test")
    assert events, "影响闸判定应留审计事件"
    assert all(e["event_type"] == "recall" for e in events)
