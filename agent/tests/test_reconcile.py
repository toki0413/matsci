"""I43: actual 必须来自真实执行/可重放经验；合成 actual 禁止."""
from __future__ import annotations

import pytest

from huginn.research.world_model_binding import (
    WorldModelPrediction,
    reconcile_prediction,
)


def _prediction() -> WorldModelPrediction:
    return WorldModelPrediction(
        prediction_id="p1",
        world_model_id="law.exoplanet",
        worldview="physics_causal",
        domain="exoplanet",
        contract_version="v1",
        predicted={"T_eq_K": 200.0},
    )


def test_actual_provenance():
    prediction = _prediction()
    for forbidden in ("", "synthetic", "fabricated", "llm", "model"):
        with pytest.raises(ValueError):
            reconcile_prediction(
                prediction, {"T_eq_K": 200.0}, actual_source=forbidden
            )

    record = reconcile_prediction(
        prediction, {"T_eq_K": 200.0}, actual_source="tool_run:42"
    )
    assert record.actual_source == "tool_run:42"
    assert record.status == "borne_out"
