"""world_state 收口件单测 — ObsVector/StateSnapshot/StateEstimator/ForwardPredictor.

覆盖: 零填充契约、状态空间/观测槽、不确定度与可辨识档位、前向预测与 [WORLD
PREDICTION]、一站入口。不依赖任何真实物理后端。
"""

from __future__ import annotations

from huginn.security import physics_schema
from huginn.security.tool_registry import tool_schema
from huginn.security.workspace import PhysicalWorkspace
from huginn.security.world_model import PhysicalAction
from huginn.security.world_state import (
    ForwardPredictor,
    ObsVector,
    StateEstimator,
    WorldStateTracker,
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


def test_world_state_tracker_step_fills_prediction_and_error():
    """用真实解析前向真值 (IdealGasWorldModel) 跑一轮, 验证 estimated + predicted + 误差."""
    from huginn.security.thermo_system import IdealGasWorldModel, ThermoExecutor

    schema = tool_schema("ideal_gas")
    wm = IdealGasWorldModel()
    tracker = WorldStateTracker(schema, wm)

    state_before = {"p": 101325.0, "V": 0.022414, "T": 273.15, "n": 1.0}
    snap = tracker.step(state_before, PhysicalAction("heat", {"dq": 100.0}))
    # 前向投影命中: predicted 含后状态的 p/T (等容加热 V 不动, T/p 升)
    assert snap.predicted is not None
    assert snap.predicted["T"] > 273.15
    assert snap.predicted["V"] == 0.022414
    # 回流: 无传感器偏置的确定性执行 → 预测 vs 实测 ≈ 0
    exe = ThermoExecutor(initial=dict(state_before))
    exe.execute(PhysicalAction("heat", {"dq": 100.0}))
    err = tracker.observe(exe.observe())
    assert err is not None
    assert err < 1e-6


def test_workspace_records_world_prediction_reflow():
    """接主循环: PhysicalWorkspace 带 schema 时逐轮记录快照 + 预测误差回流."""
    from huginn.security.thermo_system import IdealGasWorldModel, ThermoExecutor

    schema = tool_schema("ideal_gas")
    exe = ThermoExecutor()
    wa = PhysicalWorkspace(IdealGasWorldModel(), exe, schema=schema)

    wa.execute(PhysicalAction("heat", {"dq": 50.0}))
    snap = wa.last_world_snapshot()
    assert snap is not None
    assert snap.predicted is not None
    assert "p" in snap.predicted
    # 回流信号为 0 附近 (解析前向真值自洽)
    err = wa.last_prediction_error()
    assert err is not None and err < 1e-6


def test_tracker_observe_without_prediction_returns_none():
    tracker = WorldStateTracker(_SCHEMA, None)  # 无前向模型 → no prediction
    tracker.step({"p": 1.0, "V": 2.0, "T": 300.0, "n": 1.0})
    assert tracker.last_prediction is None
    assert tracker.observe({"p": 1.0}) is None


def test_prediction_error_to_reward():
    from huginn.security.world_state import prediction_error_to_reward

    assert prediction_error_to_reward(None) is None
    assert prediction_error_to_reward(float("nan")) is None
    assert prediction_error_to_reward(0.0) == 1.0
    # 单调递减: 命中(误差0)高奖励 > 小漂移 > 大漂移
    assert (
        prediction_error_to_reward(0.0)
        > prediction_error_to_reward(0.5)
        > prediction_error_to_reward(2.0)
    )
    assert 0.0 < prediction_error_to_reward(0.5) < 1.0


def test_workspace_exposes_prediction_reward():
    from huginn.security.thermo_system import IdealGasWorldModel, ThermoExecutor

    schema = tool_schema("ideal_gas")
    wa = PhysicalWorkspace(IdealGasWorldModel(), ThermoExecutor(), schema=schema)
    wa.execute(PhysicalAction("heat", {"dq": 50.0}))
    reward = wa.last_prediction_reward()
    assert reward is not None
    assert reward > 0.99  # 解析前向真值自洽 → 误差≈0 → 命中奖励≈1


def test_world_prediction_reward_reflows_to_bandit_and_episodic(tmp_path):
    """阶段2 回流: 世界预测命中奖励 → bandit r_phys + episodic episode 节点."""
    import os

    import huginn.autoloop.bandit as bd
    from huginn.kg import ProjectKnowledgeGraph
    from huginn.security.thermo_system import IdealGasWorldModel, ThermoExecutor

    os.environ["HUGINN_CACHE_DIR"] = str(tmp_path)
    bd.WorkflowBandit._instance = None
    bd.VariantArchive._instance = None

    schema = tool_schema("ideal_gas")
    wa = PhysicalWorkspace(IdealGasWorldModel(), ThermoExecutor(), schema=schema)
    wa.execute(PhysicalAction("heat", {"dq": 100.0}))
    reward = wa.last_prediction_reward()
    assert reward is not None and reward > 0.99

    # bandit: r_phys 吃世界预测命中奖励
    bandit = bd.WorkflowBandit.get_instance()
    bandit.record_variant_outcome("test_v1", "obj_world", success=True, r_phys=reward)
    belief = bandit.get_belief("test_v1", "obj_world")
    assert belief is not None
    assert belief.last_r_phys == reward

    # episodic: episode 节点记录预测命中
    kg = ProjectKnowledgeGraph(tmp_path / "kg")
    nid = kg.add_episode_node(1, "heat", f"prediction reward={reward:.3f}", "success")
    assert nid == "episode_1"
