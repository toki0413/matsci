"""漂移事件账测试：世界模型 prediction_error 落库、等级过滤、租户隔离."""
from __future__ import annotations

from huginn.metacog.drift_ledger import (
    DriftEvent,
    DriftLedger,
    record_prediction_error,
    record_prediction_error_value,
)
from huginn.research.world_model_binding import (
    WorldModelPrediction,
    reconcile_prediction,
)


def _ledger(tmp_path) -> DriftLedger:
    return DriftLedger(tmp_path / "drift.db")


def test_drift_event_roundtrip(tmp_path):
    ledger = _ledger(tmp_path)
    event = DriftEvent(
        tenant_id="t1",
        session_id="s1",
        detector="propensity",
        baseline_id="b1",
        observed=0.5,
        threshold=0.15,
        level="review",
        evidence={"x": 1},
    )
    ledger.record(event)
    events = ledger.list_events(tenant_id="t1", session_id="s1")
    assert len(events) == 1
    assert events[0]["detector"] == "propensity"
    assert events[0]["evidence_json"] == {"x": 1}


def test_tenant_isolation(tmp_path):
    ledger = _ledger(tmp_path)
    ledger.record(
        DriftEvent(
            tenant_id="t1",
            session_id="s1",
            detector="propensity",
            baseline_id="b1",
            observed=0.5,
            threshold=0.15,
            level="review",
        )
    )
    assert ledger.list_events(tenant_id="t2") == []


def test_prediction_error_writes_review_and_intervene(tmp_path):
    ledger = _ledger(tmp_path)
    prediction = WorldModelPrediction(
        prediction_id="p1",
        world_model_id="law.exoplanet",
        worldview="physics_causal",
        domain="exoplanet",
        contract_version="v1",
        predicted={"T_eq_K": 200.0},
    )
    ok = reconcile_prediction(
        prediction, {"T_eq_K": 200.0}, actual_source="tool_run:1",
        metrics=("T_eq_K",),
    )
    assert record_prediction_error(
        ok, tenant_id="t1", session_id="s1", ledger=ledger
    ) is None

    bad = reconcile_prediction(
        prediction, {"T_eq_K": 240.0}, actual_source="tool_run:1",
        metrics=("T_eq_K",),
    )
    event_id = record_prediction_error(
        bad, tenant_id="t1", session_id="s1", ledger=ledger
    )
    assert event_id
    events = ledger.list_events(tenant_id="t1", session_id="s1")
    assert len(events) == 1
    assert events[0]["detector"] == "prediction_error"
    assert events[0]["level"] in ("review", "intervene")


def test_record_prediction_error_value_levels(tmp_path):
    ledger = _ledger(tmp_path)
    assert record_prediction_error_value(
        0.01, tenant_id="t1", session_id="s1", ledger=ledger
    ) is None
    review = record_prediction_error_value(
        0.10, tenant_id="t1", session_id="s1", ledger=ledger
    )
    intervene = record_prediction_error_value(
        0.30, tenant_id="t1", session_id="s1", ledger=ledger
    )
    assert review and intervene
    levels = [
        e["level"] for e in ledger.list_events(tenant_id="t1", session_id="s1")
    ]
    assert levels == ["review", "intervene"]
