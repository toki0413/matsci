"""VIGIL 降级状态机与漂移检测测试."""
from __future__ import annotations

from huginn.metacog.drift_gate import DriftThresholds, evaluate_drift
from huginn.metacog.vigil import MonitorState, VigilMonitor


def test_monitor_starts_full_and_degrades_on_error_rate():
    m = VigilMonitor("m1", window=5, error_threshold=0.2)
    assert m.status().state == "L0"
    assert m.status().trust == "full"
    for _ in range(2):
        m.record_job(False)
    status = m.status()
    assert status.state == "L1"
    assert status.trust == "partial"
    assert m.degradation_events


def test_all_detectors_down_goes_blind():
    m = VigilMonitor("m1")
    m.record_detector("structure", False)
    m.record_detector("demand", False)
    m.record_detector("default", False)
    assert m.status().state == "L2"
    assert m.status().trust == "blind"
    assert m.submit("job") == "DROPPED"


def test_heartbeat_timeout_goes_offline():
    m = VigilMonitor("m1", heartbeat_timeout_s=10)
    m.heartbeat("2026-01-01T00:00:00+00:00")
    status = m.tick("2026-01-01T00:00:20+00:00")
    assert status.state == "L3"
    assert status.trust == "unknown"


def test_quarantine_and_bounded_restart():
    m = VigilMonitor("m1", max_restarts=2, restart_window_s=600)
    m.quarantine("runaway")
    assert m.status().state == "L4"
    assert m.restart("2026-01-01T00:00:00+00:00") is True
    assert m.status().state == "L1"
    m.quarantine("runaway")
    assert m.restart("2026-01-01T00:00:10+00:00") is True
    m.quarantine("runaway")
    assert m.restart("2026-01-01T00:00:20+00:00") is False
    assert m.status().state == "L3"


def test_queue_is_bounded_and_never_backpressures():
    m = VigilMonitor("m1", max_queue=2)
    assert m.submit("a") == "a"
    assert m.submit("b") == "b"
    assert m.submit("c") == "DROPPED"
    assert m.pop_job() == "a"


def test_status_never_raises():
    m = VigilMonitor("m1")
    m._state = MonitorState.L0
    assert m.status().state == "L0"


def test_drift_levels_are_monotonic():
    signals = evaluate_drift(
        propensity=(1.0, 0.8),
        truth_seeking=(1.0, 0.5),
        causal_edit_distance=0.5,
        thresholds=DriftThresholds(),
    )
    levels = [s.level for s in signals]
    assert levels[0] == "pause", "最严重信号排第一"
    assert "intervene" in levels
    assert "review" in levels
    assert all(s.detector in {"propensity", "truth_seeking", "causal_edit"} for s in signals)


def test_drift_below_thresholds_is_empty():
    assert evaluate_drift(
        propensity=(1.0, 0.95),
        truth_seeking=(1.0, 0.90),
        causal_edit_distance=0.1,
    ) == []
