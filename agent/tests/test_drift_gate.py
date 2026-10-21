"""I29/I46: 告警单调升级、pause 不可被任务 Agent 自清；预测误差落 drift_events."""
from __future__ import annotations

from huginn.metacog.drift_gate import DriftGate, evaluate_drift
from huginn.metacog.drift_ledger import DriftLedger, record_prediction_error
from huginn.research.world_model_binding import (
    WorldModelPrediction,
    reconcile_prediction,
)


def test_pause_not_self_clearable():
    gate = DriftGate()
    signals = evaluate_drift(
        propensity=(1.0, 0.5),
        truth_seeking=(1.0, 0.5),
        causal_edit_distance=0.9,
    )
    gate.update(signals)
    assert gate.level == "pause"
    assert gate.can_execute() is False

    assert gate.clear(actor="agent", confirmation=True) is False
    assert gate.level == "pause", "任务 Agent 不得自清 pause"
    assert gate.clear(actor="task", confirmation=True) is False

    assert gate.clear(actor="user", confirmation=False) is False
    assert gate.clear(actor="user", confirmation=True) is True
    assert gate.level == "ok"
    assert gate.can_execute() is True


def test_alerts_escalate_monotonically():
    gate = DriftGate()
    gate.update(evaluate_drift(propensity=(1.0, 0.8)))  # review
    assert gate.level == "review"
    gate.update(evaluate_drift(truth_seeking=(1.0, 0.5)))  # intervene
    assert gate.level == "intervene"
    gate.update([])  # 无信号不降级
    assert gate.level == "intervene"


def test_prediction_error_signal(tmp_path):
    """I46: prediction_error 超阈值必须落 drift_events，禁止静默."""
    prediction = WorldModelPrediction(
        prediction_id="p1",
        world_model_id="law.exoplanet",
        worldview="physics_causal",
        domain="exoplanet",
        contract_version="v1",
        predicted={"T_eq_K": 200.0},
    )
    record = reconcile_prediction(
        prediction,
        {"T_eq_K": 400.0},
        actual_source="tool_run:1",
    )
    assert record.status == "falsified"

    ledger = DriftLedger(tmp_path / "drift.db")
    event_id = record_prediction_error(
        record, tenant_id="t1", session_id="s1", ledger=ledger
    )
    assert event_id
    events = ledger.list_events(tenant_id="t1", session_id="s1")
    assert events
    assert events[0]["detector"] == "prediction_error"
    assert events[0]["level"] in ("review", "intervene")
