"""VIGIL 降级状态机与漂移检测测试."""
from __future__ import annotations

from huginn.metacog.drift_gate import DriftThresholds, evaluate_drift
from huginn.metacog.drift_ledger import DriftLedger
from huginn.metacog.vigil import MonitorState, VigilMonitor
from huginn.metacog.vigil_runtime import VigilRuntime


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


def test_no_task_authority(monkeypatch):
    """I25: 监控环不执行用户任务、不写 authorize、不改 PersonaKernel."""
    from huginn.normative.ledger import NormativeLedger

    def _boom(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("监控不得写规范账")

    monkeypatch.setattr(NormativeLedger, "append_act", _boom)
    runtime = VigilRuntime(monitor=VigilMonitor("v1"), enabled=True)
    runtime.run_once("2026-01-01T00:00:00+00:00")  # 不触发 append
    for forbidden in ("authorize", "execute", "commit_act", "revoke"):
        assert not hasattr(runtime, forbidden)


def test_protected_surface():
    """I28: 监控更新只作用于诊断/自适应面，内核不可变."""
    runtime = VigilRuntime(monitor=VigilMonitor("v1"), enabled=True)
    for forbidden in ("set_persona", "apply_kernel_delta", "modify_persona_kernel"):
        assert not hasattr(runtime, forbidden)
    assert hasattr(runtime, "record_signals"), "监控只能写漂移诊断"


def test_explicit_degradation():
    """I33: 监控自身失败必须显式降级（精确错误 + unknown），禁止静默通过."""
    class _Broken:
        def status(self):
            raise RuntimeError("boom")

    runtime = VigilRuntime(monitor=_Broken(), enabled=False)  # type: ignore[arg-type]
    status = runtime.status()
    assert status.state == "L4"
    assert status.trust == "unknown"
    assert status.degraded_reason == "status_failure"


def test_runtime_records_jobs_and_heartbeat():
    runtime = VigilRuntime(
        monitor=VigilMonitor("v1", window=5, error_threshold=0.2),
        enabled=True,
    )
    runtime.run_once("2026-01-01T00:00:00+00:00")
    runtime.record_job(False)
    runtime.record_job(False)
    assert runtime.status().state == "L1"


def test_runtime_disabled_does_not_start_thread():
    runtime = VigilRuntime(enabled=False)
    runtime.start()
    assert runtime._thread is None
    runtime.stop()


def test_runtime_persists_drift_signals(tmp_path):
    ledger = DriftLedger(tmp_path / "drift.db")
    runtime = VigilRuntime(enabled=True)
    signals = evaluate_drift(
        propensity=(1.0, 0.5),
        truth_seeking=(1.0, 0.5),
        causal_edit_distance=0.5,
    )
    ids = runtime.record_signals(
        signals, tenant_id="t1", session_id="s1", ledger=ledger
    )
    assert len(ids) == len(signals)
    events = ledger.list_events(tenant_id="t1", session_id="s1")
    assert len(events) == len(signals)
    assert {e["level"] for e in events} >= {"review", "intervene", "pause"}
