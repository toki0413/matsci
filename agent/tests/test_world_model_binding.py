"""世界模型 binding 测试：预测不直接成事实、对账升格、worldview 隔离、漂移信号."""
from __future__ import annotations

import pytest

from huginn.research.world_model_binding import (
    WorldModelPrediction,
    authority_for,
    prediction_error_signal,
    promotion_status,
    reconcile_prediction,
    to_pmk_stance,
    worldview_compatible,
)


def _prediction(**overrides) -> WorldModelPrediction:
    data = {
        "prediction_id": "p1",
        "world_model_id": "law.exoplanet",
        "worldview": "physics_causal",
        "domain": "exoplanet",
        "contract_version": "v1",
        "predicted": {"T_eq_K": 200.0, "S_Wm2": 1000.0},
    }
    data.update(overrides)
    return WorldModelPrediction(**data)


def test_prediction_defaults_to_low_evidential_authority():
    authority = authority_for(None)
    assert authority.evidential_authority <= 0.2
    assert authority.influence_authority <= 0.4


def test_reconcile_requires_real_actual_source():
    prediction = _prediction()
    with pytest.raises(ValueError):
        reconcile_prediction(prediction, {"T_eq_K": 200.0}, actual_source="")
    with pytest.raises(ValueError):
        reconcile_prediction(
            prediction, {"T_eq_K": 200.0}, actual_source="synthetic"
        )


def test_reconcile_three_states_and_promotion():
    prediction = _prediction()
    ok = reconcile_prediction(
        prediction,
        {"T_eq_K": 200.0, "S_Wm2": 1000.0},
        actual_source="tool_run:1",
    )
    assert ok.status == "borne_out"
    assert promotion_status(ok) == "reconciled"
    assert authority_for(ok).evidential_authority == 0.6
    assert authority_for(ok, independent_runs=1).evidential_authority == 0.9

    bad = reconcile_prediction(
        prediction,
        {"T_eq_K": 300.0},
        actual_source="tool_run:1",
    )
    assert bad.status == "falsified"
    assert promotion_status(bad) == "refuted"
    assert authority_for(bad).evidential_authority == 0.1

    insufficient = reconcile_prediction(
        prediction,
        {},
        actual_source="tool_run:1",
    )
    assert insufficient.status == "insufficient"
    assert promotion_status(insufficient) == "insufficient"


def test_worldview_must_match():
    a = _prediction()
    b = _prediction(prediction_id="p2", worldview="latent_transition")
    assert not worldview_compatible(a, b)
    assert worldview_compatible(a, _prediction(prediction_id="p3"))


def test_prediction_error_signal_levels():
    prediction = _prediction()
    record = reconcile_prediction(
        prediction,
        {"T_eq_K": 200.0},
        actual_source="tool_run:1",
        metrics=("T_eq_K",),
    )
    signal = prediction_error_signal(record)
    assert signal["detector"] == "prediction_error"
    assert signal["level"] == "ok"

    bad = reconcile_prediction(
        prediction,
        {"T_eq_K": 240.0},
        actual_source="tool_run:1",
        metrics=("T_eq_K",),
    )
    signal_bad = prediction_error_signal(bad)
    assert signal_bad["level"] in ("review", "intervene")


def test_pmk_stance_is_labeled_not_fact():
    stance = to_pmk_stance(_prediction())
    assert stance["source"] == "world_model"
    assert stance["worldview"] == "physics_causal"
    assert stance["domain"] == "exoplanet"
    assert "physics_causal" in stance["text"]
