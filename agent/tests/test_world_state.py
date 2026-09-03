"""world_state 收口件单测 — ObsVector/StateSnapshot/StateEstimator/ForwardPredictor.

覆盖: 零填充契约、状态空间/观测槽、不确定度与可辨识档位、前向预测与 [WORLD
PREDICTION]、一站入口。不依赖任何真实物理后端。
"""

from __future__ import annotations

from huginn.security import physics_schema
from huginn.security.world_state import (
    ForwardPredictor,
    ObsVector,
    StateEstimator,
    snapshot_from_schema,
)

_SCHEMA = {
    "space": {"state": ["p", "V", "T", "n"], "action": ["heat", "move"]},
    "observables": ["p", "T"],
    "contract_version": physics_schema.SHARED_CONTRACT_VERSION,
}


def test_obsvector_zero_pads_missing_slots():
    ov = ObsVector.from_schema(_SCHEMA)
    assert ov.slots == ("p", "V", "T", "n")
    assert ov.values == (0.0, 0.0, 0.0, 0.0)
    assert ov.contract_version == physics_schema.SHARED_CONTRACT_VERSION
    # 只给 p, 其余零填充; 未登记键忽略
    packed = ov.pack({"p": 1.5, "V": 2.5, "surprise": 99.0})
    assert packed.values == (1.5, 2.5, 0.0, 0.0)


def test_obsvector_as_dict_maps_slots():
    ov = ObsVector.from_schema(_SCHEMA).pack({"p": 3.0})
    assert ov.as_dict() == {"p": 3.0, "V": 0.0, "T": 0.0, "n": 0.0}


def test_state_estimator_uncertainty_and_level():
    est = StateEstimator(_SCHEMA)
    snap = est.estimate({"p": 1.0, "V": 5.0, "T": 300.0, "n": 2.0})
    assert snap.state == {"p": 1.0, "V": 5.0, "T": 300.0, "n": 2.0}
    assert snap.observables == ("p", "T")
    # 观测项给 σ, 不可估项给 None
    assert snap.uncertainty["p"] == 0.01
    assert snap.uncertainty["V"] is None
    # 4 状态里观测 2 → limited
    assert snap.identifiable_level == "limited"


def test_state_estimator_deficient_when_no_observables():
    est = StateEstimator({"space": {"state": ["p", "V"]}, "observables": []})
    snap = est.estimate({"p": 1.0, "V": 2.0})
    assert snap.identifiable_level == "deficient"


def test_forward_predictor_with_predictor():
    fp = ForwardPredictor(_SCHEMA)
    snap = StateEstimator(_SCHEMA).estimate({"p": 1.0, "V": 5.0, "T": 300.0, "n": 2.0})

    class _Fake:
        def predict(self, state):
            return {"p": state["p"] * 2.0}

    pred = fp.predict(snap, predictor=_Fake())
    assert pred == {"p": 2.0}
    assert "[WORLD PREDICTION]" in fp.to_prompt(pred)
    # 无 predictor → None, prompt 为空
    assert fp.predict(snap, predictor=None) is None
    assert fp.to_prompt(None) == ""


def test_forward_predictor_tolerates_action_signature():
    fp = ForwardPredictor(_SCHEMA)
    snap = StateEstimator(_SCHEMA).estimate({"p": 1.0})
    called = {"n": 0}

    def _predict(state, action):  # noqa: ANN001
        called["n"] += 1
        return {"n": 1}

    fp.predict(snap, predictor=_FakeWithArgs(_predict), action="heat")
    assert called["n"] == 1


class _FakeWithArgs:  # noqa: D401
    def __init__(self, fn):
        self._fn = fn

    def predict(self, state, action):  # noqa: ANN001
        return self._fn(state, action)


def test_snapshot_from_schema_standalone_entry():
    snap = snapshot_from_schema(_SCHEMA, {"p": 1.0, "T": 280.0})
    assert snap.state["p"] == 1.0
    assert snap.identifiable_level == "limited"
    assert "[STATE]" in snap.to_prompt()
