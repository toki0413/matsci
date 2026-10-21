"""I6/I7: 认知账——取代保留双份、按 valid-time 检索、冲突可寻址."""
from __future__ import annotations

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


def test_tenant_isolation(tmp_path):
    ledger = BeliefLedger(tmp_path / "belief.db")
    ledger.append(_record(tenant_id="t1"))
    assert ledger.get("b1", tenant_id="t2") is None
    assert ledger.current("t2", "band_gap") is None
    assert ledger.list_records("t2") == []
