"""行为指纹：工具族分布、风险分布与 JS 散度可测、可比较."""
from __future__ import annotations

from huginn.research.behavior_fingerprint import (
    BehaviorMonitor,
    behavior_drift,
    fingerprint_from_events,
    js_divergence,
)


def _events(tool: str, n: int = 1, **overrides) -> list[dict]:
    return [{"tool": tool, **overrides} for _ in range(n)]


def test_fingerprint_distributions_and_js():
    baseline = fingerprint_from_events(
        _events("vasp_tool", 3) + _events("lammps_tool", 1),
        persona_id="p1",
    )
    assert baseline.n_events == 4
    assert abs(sum(baseline.tool_family_dist.values()) - 1.0) < 1e-9
    assert baseline.tool_family_dist["vasp"] == 0.75

    identical = fingerprint_from_events(
        _events("vasp_tool", 3) + _events("lammps_tool", 1),
        persona_id="p1",
    )
    assert js_divergence(
        baseline.tool_family_dist, identical.tool_family_dist
    ) == 0.0

    shifted = fingerprint_from_events(
        _events("lammps_tool", 4), persona_id="p1"
    )
    divergence = js_divergence(
        baseline.tool_family_dist, shifted.tool_family_dist
    )
    assert 0.0 < divergence <= 1.0

    drift = behavior_drift(baseline, shifted)
    assert drift["tool_family_js"] > 0
    assert drift["n_events"] == 4


def test_empty_events_are_zero_not_fabricated():
    empty = fingerprint_from_events([], persona_id="p1")
    assert empty.n_events == 0
    assert empty.tool_family_dist == {}
    assert js_divergence({}, {}) == 0.0


def test_monitor_baseline_and_drift_event(tmp_path):
    from huginn.metacog.drift_ledger import DriftLedger

    ledger = DriftLedger(tmp_path / "drift.db")
    monitor = BehaviorMonitor(window=10, threshold=0.1)
    for _ in range(5):
        monitor.record_event(
            tenant_id="t1", persona_id="p1",
            event={"tool": "vasp_tool", "risk": "low"},
        )
    assert monitor.maybe_record_drift(
        tenant_id="t1", persona_id="p1", ledger=ledger
    ) is None, "首次只建基线"

    # 窗口滚动为全 lammps → JS=1 > 阈值
    for _ in range(10):
        monitor.record_event(
            tenant_id="t1", persona_id="p1",
            event={"tool": "lammps_tool", "risk": "high"},
        )
    event_id = monitor.maybe_record_drift(
        tenant_id="t1", persona_id="p1", ledger=ledger
    )
    assert event_id
    events = ledger.list_events(tenant_id="t1", session_id="p1")
    assert events[0]["detector"] == "behavior_fingerprint"
    assert events[0]["level"] == "review"

    # 其他租户不受影响
    assert monitor.current(tenant_id="t2", persona_id="p1").n_events == 0


def test_rates_and_boundary_hits():
    events = [
        {"tool": "vasp_tool", "clarified": True, "evidence_gate_passed": True},
        {"tool": "vasp_tool", "clarified": False, "evidence_gate_passed": False},
        {"tool": "vasp_tool", "binding_hit": True, "boundary_hit": True},
    ]
    fp = fingerprint_from_events(events, persona_id="p1")
    assert fp.clarification_rate == 1 / 3
    assert fp.evidence_gate_pass_rate == 1 / 3
    assert fp.authorization_boundary_hits == 1
