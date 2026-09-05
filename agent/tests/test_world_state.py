"""world_state 收口件单测 — ObsVector/StateSnapshot/StateEstimator/ForwardPredictor.

覆盖: 零填充契约、状态空间/观测槽、不确定度与可辨识档位、前向预测与 [WORLD
PREDICTION]、一站入口。不依赖任何真实物理后端。
"""

from __future__ import annotations

import time

import pytest

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


def test_module_reconcile_r_phys_folds_or_passthrough():
    from huginn.security.world_state import reconcile_r_phys

    # 有 world_reward → 各 0.5 并进
    assert reconcile_r_phys(0.6, 1.0) == pytest.approx(0.8)
    # 无 world_reward (None) → 原样返回 base, 不扣分
    assert reconcile_r_phys(0.55, None) == 0.55
    # graft 可调
    assert reconcile_r_phys(0.4, 1.0, graft=0.25) == pytest.approx(0.85)


def test_world_catalog_block_lists_registered_tools():
    """autoloop 注入: 世界模型注册表列出 ToolSpec 解析前向真值 (advisory, 零 LLM)."""
    from huginn.autoloop.engine_observe import EngineObserveMixin
    from huginn.security.tool_registry import registered_tools

    class _Dummy(EngineObserveMixin):
        pass

    block = _Dummy()._build_world_catalog_block()
    assert "已注册解析世界模型" in block
    assert "ideal_gas" in block
    assert "observables=[p,T]" in block
    assert len(registered_tools()) >= 4


def test_world_model_block_includes_catalog_when_no_memory():
    """无长程记忆时仍注入解析世界模型注册表, 不因缺相似历史而空转."""
    from huginn.autoloop.engine_observe import EngineObserveMixin

    class _Dummy(EngineObserveMixin):
        pass

    block = _Dummy()._build_world_model_block("hypothesis test")
    assert "已注册解析世界模型" in block


def test_sparse_reward_tuner_learns_sensor_bias():
    """深度 RL 奖励: 预测命中奖励驱动的梯度学习能把 sensor 系统偏置学出来."""
    from huginn.security.world_state import SparseRewardTuner

    tuner = SparseRewardTuner(["T"], lr=0.02)
    ideal = 300.0
    observed = ideal + 7.0  # 温度表 +7K 系统偏置 (未知)
    for _ in range(3000):
        tuner.update({"T": ideal}, {"T": observed})
    # 学到的加性残差逼近真实偏置
    assert tuner.offset["T"] == pytest.approx(7.0, abs=1e-3)
    # 收敛后命中奖励接近 1 (预测>>实测误差趋 0)
    r = tuner.update({"T": ideal}, {"T": observed})
    assert r is not None and r > 0.99


def test_learnable_forward_model_learns_transfer():
    """P2 surrogate: 数据驱动线性前向代理学习状态转移 (全状态变化保证满秩)."""
    import random

    from huginn.security.thermo_system import _R, forward
    from huginn.security.world_model import PhysicalAction
    from huginn.security.world_state import LearnableForwardModel

    schema = tool_schema("ideal_gas")
    sur = LearnableForwardModel(["p", "V", "T", "n"])
    rng = random.Random(0)
    for _ in range(120):
        T = rng.uniform(260, 340)
        V = rng.uniform(0.015, 0.03)
        n = rng.uniform(0.8, 1.2)
        s = {"p": _R * T * n / V, "V": V, "T": T, "n": n}
        a_heat = PhysicalAction("heat", {"dq": rng.uniform(-50, 50)})
        sur.fit(s, a_heat, forward(s, a_heat))
        a_move = PhysicalAction("move", {"v": rng.uniform(0.015, 0.03)})
        sur.fit(s, a_move, forward(s, a_move))
    s0 = {"p": _R * 310.0 * 1.0 / 0.02, "V": 0.02, "T": 310.0, "n": 1.0}
    # move: V_after = 目标 v, 线性代理精确够到
    pred_v = sur.predict(s0, PhysicalAction("move", {"v": 0.025}))
    assert pred_v is not None and "V" in pred_v
    assert pred_v["V"] == pytest.approx(0.025, rel=1e-6)
    # heat: 温度转移可学, 给出非空预测
    pred_h = sur.predict(s0, PhysicalAction("heat", {"dq": 20.0}))
    assert pred_h is not None and "T" in pred_h
    # 未见过的动作类型 → None (不硬编造)
    assert sur.predict(s0, PhysicalAction("kick", {"dv": 0.1})) is None


def test_learnable_surrogate_drops_into_forward_predictor():
    """P2 surrogate 契约: 可直接当 ForwardPredictor.predictor 替换解析真值."""
    import random

    from huginn.security.thermo_system import _R, forward
    from huginn.security.world_model import PhysicalAction
    from huginn.security.world_state import (
        ForwardPredictor,
        LearnableForwardModel,
        StateEstimator,
    )

    schema = tool_schema("ideal_gas")
    sur = LearnableForwardModel(["p", "V", "T", "n"])
    rng = random.Random(2)
    for _ in range(120):
        T = rng.uniform(260, 340)
        V = rng.uniform(0.015, 0.03)
        n = rng.uniform(0.8, 1.2)
        s = {"p": _R * T * n / V, "V": V, "T": T, "n": n}
        a_move = PhysicalAction("move", {"v": rng.uniform(0.015, 0.03)})
        sur.fit(s, a_move, forward(s, a_move))
    s0 = {"p": _R * 300.0 / 0.02, "V": 0.02, "T": 300.0, "n": 1.0}
    snap = StateEstimator(schema).estimate(s0)
    pred = ForwardPredictor(schema).predict(
        snap, predictor=sur, action=PhysicalAction("move", {"v": 0.023})
    )
    assert pred is not None and "V" in pred
    assert pred["V"] == pytest.approx(0.023, rel=1e-6)


def test_tracker_observe_feeds_learnable_surrogate():
    """P2 线: observe 把本轮 (前状态, 动作, 实测后状态) 喂给学代理积累真实运行数据.

    用活着的主循环跑多轮: tracker.step → tracker.observe(实测后状态), surrogate 通过
    observe 内部自学习 → 能对同型动作输出预测 (不再硬编造 None). 喂养独立于预测成败
    (世界模型传 None → 无预测, observe 返回 None, 但代理仍照常积累样本).
    """
    import random

    from huginn.security.thermo_system import _R, forward
    from huginn.security.world_state import LearnableForwardModel

    schema = tool_schema("ideal_gas")
    sur = LearnableForwardModel(["p", "V", "T", "n"])
    # 世界模型传 None: 无预测 → observe 返回 None, 但代理喂养不受影响
    tracker = WorldStateTracker(schema, None, surrogate=sur)
    rng = random.Random(3)

    for _ in range(60):
        T = rng.uniform(260, 340)
        V = rng.uniform(0.015, 0.03)
        n = rng.uniform(0.8, 1.2)
        s = {"p": _R * T * n / V, "V": V, "T": T, "n": n}
        a = PhysicalAction("move", {"v": rng.uniform(0.015, 0.03)})
        tracker.step(s, a)
        assert tracker.observe(forward(s, a)) is None  # 无预测 → None
    # 同型 move 动作已有 60 条样本 → 回学出可学传输 → 输出预测而非硬编造
    pred = sur.predict(
        {"p": _R * 300.0 / 0.02, "V": 0.02, "T": 300.0, "n": 1.0},
        PhysicalAction("move", {"v": 0.023}),
    )
    assert pred is not None and "V" in pred
    assert pred["V"] == pytest.approx(0.023, rel=1e-6)


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


def test_workspace_prediction_reward_avg_over_steps():
    """阶段3: 多步运行平均预测命中奖励 — runner 求整协议 r_phys 贡献."""
    from huginn.security.thermo_system import IdealGasWorldModel, ThermoExecutor

    schema = tool_schema("ideal_gas")
    wa = PhysicalWorkspace(IdealGasWorldModel(), ThermoExecutor(), schema=schema)
    for dq in (50.0, 100.0, 150.0):
        wa.execute(PhysicalAction("heat", {"dq": dq}))
    avg = wa.prediction_reward_avg()
    assert avg is not None
    assert 0.99 < avg <= 1.0  # 解析前向真值自洽 → 每步命中奖励≈1


def test_workspace_reconcile_r_phys_folds_prediction_reward():
    """阶段3: runner 默认消费 — reconcile_r_phys 把预测命中奖励并进底层 r_phys."""
    from huginn.security.thermo_system import IdealGasWorldModel, ThermoExecutor

    schema = tool_schema("ideal_gas")
    wa = PhysicalWorkspace(IdealGasWorldModel(), ThermoExecutor(), schema=schema)
    wa.execute(PhysicalAction("heat", {"dq": 50.0}))

    reconciled = wa.reconcile_r_phys(0.6)
    # 预测命中奖励≈1, base=0.6 → 0.5*0.6 + 0.5*1 ≈ 0.8
    assert reconciled == pytest.approx(0.8, abs=0.02)
    # 无 schema/无世界跟踪 → 原样返回 base, 不因缺世界模型扣分
    wa2 = PhysicalWorkspace(IdealGasWorldModel(), ThermoExecutor())
    wa2.execute(PhysicalAction("heat", {"dq": 50.0}))
    assert wa2.reconcile_r_phys(0.55) == 0.55


def test_workspace_execute_feeds_learnable_surrogate():
    """P2 线: workspace 带 surrogate 时, 真实 execute (step+observe) 把观测对喂给学代理.

    用活着的主循环跑多次 heat, 代理从实际执行的 (前状态, 动作, 实测后状态) 积累样本,
    独立于解析前向真值 → 能对同型 heat 动作输出非空预测 (不在解析循环里冒充新物理).
    """
    import random

    from huginn.security.thermo_system import IdealGasWorldModel, ThermoExecutor
    from huginn.security.world_state import LearnableForwardModel

    schema = tool_schema("ideal_gas")
    sur = LearnableForwardModel(["p", "V", "T", "n"])
    wa = PhysicalWorkspace(
        IdealGasWorldModel(), ThermoExecutor(), schema=schema, surrogate=sur
    )
    rng = random.Random(5)
    for _ in range(60):
        wa.execute(PhysicalAction("heat", {"dq": rng.uniform(-50, 50)}))

    # surrogate 已积累样本 → heat 动作能回归出非空预测
    pred = sur.predict({"p": 101325.0, "V": 0.022414, "T": 273.15, "n": 1.0},
                       PhysicalAction("heat", {"dq": 10.0}))
    assert pred is not None
    assert "T" in pred
    # 世界跟踪本体照常工作 (解析前向真值自洽)
    assert wa.reconcile_r_phys(0.6) == pytest.approx(0.8, abs=0.05)


# ── 阶段7: 最小贝叶斯滤波 (Life Operators Eq.4) ────────────────
def test_gaussian_update_folds_prior_and_observation():
    """单键高斯折叠: 先验(0,1) × 观测(2,1) → 后验均值 1, 方差 0.5 (K=0.5)."""
    from huginn.security.world_state import _gaussian_update

    mu, var = _gaussian_update(0.0, 1.0, 2.0, 1.0)
    assert mu == pytest.approx(1.0, rel=1e-9)
    assert var == pytest.approx(0.5, rel=1e-9)
    # 仅观测 → 直接取观测 (无先验可折叠)
    mu2, var2 = _gaussian_update(None, None, 5.0, 2.0)
    assert mu2 == 5.0 and var2 == 2.0
    # 无观测 → 保留先验
    mu3, var3 = _gaussian_update(3.0, 0.4, None, None)
    assert mu3 == 3.0 and var3 == 0.4


def test_state_estimator_filter_folds_predict_and_observation():
    """filter: 先验(0,1) —预测→1—观测→3 三层折叠 → 后验 ≈(1.333, 1/3)."""
    from huginn.security.world_state import StateSnapshot

    est = StateEstimator(_SCHEMA)
    prior = StateSnapshot(
        state={"p": 0.0, "V": 5.0, "T": 300.0, "n": 2.0},
        posterior_var={"p": 1.0, "V": None, "T": 1.0, "n": None},
    )
    snap = est.filter(prior, predicted={"p": 1.0}, observation={"p": 3.0}, obs_sigma=1.0)
    # p: 先验 0 → 预测 (K=0.5 → 0.5) → 观测 3 (K=1/3 → 0.5 + 2/3·(3−0.5)≈1.333)
    assert snap.state["p"] == pytest.approx(1.333, rel=1e-2)
    assert snap.posterior_var["p"] == pytest.approx(1.0 / 3.0, rel=1e-2)
    # V/n 无预测无观测 → 保持原值
    assert snap.state["V"] == 5.0 and snap.state["n"] == 2.0
    # 观测噪声大 → 后验被拉回观测较近 (体现"不确定性按方差加权")
    snap2 = est.filter(
        prior, predicted={"p": 1.0}, observation={"p": 3.0}, obs_sigma=10.0
    )
    assert snap2.state["p"] < snap.state["p"]  # 观测更不可信 → 后验更贴近预测


def test_filter_via_prior_uncertainty_sigma_interpreted_as_sigma2():
    """从 estimate 出的 σ(uncertainty) 平滑进入滤波: p 观测 σ=0.01 → 后验方差 σ²缩小."""
    est = StateEstimator(_SCHEMA)
    prior = est.estimate({"p": 1.0, "V": 5.0, "T": 300.0, "n": 2.0})
    snap = est.filter(prior, predicted={"p": 1.0}, observation={"p": 1.02})
    # p 强观测先验 → 两次 Gauss 折叠后 posterior_var 应小于初始 σ²=1e-4 (不确定性收敛)
    assert snap.posterior_var["p"] is not None
    assert 0 < snap.posterior_var["p"] < 1e-4


def test_tracker_observe_produces_bayesian_posterior():
    """阶段7 接线: WorldStateTracker.observe 用演化先验 × 实测做滤波 → last_posterior."""
    from huginn.security.thermo_system import IdealGasWorldModel, ThermoExecutor

    schema = tool_schema("ideal_gas")
    wm = IdealGasWorldModel()
    tracker = WorldStateTracker(schema, wm)
    sb = {"p": 101325.0, "V": 0.022414, "T": 273.15, "n": 1.0}
    tracker.step(sb, PhysicalAction("heat", {"dq": 100.0}))
    exe = ThermoExecutor(initial=dict(sb))
    exe.execute(PhysicalAction("heat", {"dq": 100.0}))
    tracker.observe(exe.observe())
    assert tracker.last_posterior is not None
    # V 等容不可直测 → 由前向预测/观测折叠后仍有有限后验方差 (σ 远小于 1)
    assert tracker.last_posterior.posterior_var.get("V") is not None
    assert tracker.last_posterior.posterior_var["V"] < 1e-3


# ── token 瘦身: to_prompt 只带结论 ─────────────────────────────
def _dummy_mixin():
    from huginn.autoloop.engine_observe import EngineObserveMixin

    class _Dummy(EngineObserveMixin):
        pass

    return _Dummy()


def test_snapshot_to_prompt_keeps_identifiable_and_diff():
    """to_prompt 精简: 只列可辨识 σ 与有变化的 Δpred, 无变化标 Δpred=0."""
    est = StateEstimator(_SCHEMA)
    snap = est.estimate({"p": 1.0, "V": 5.0, "T": 300.0, "n": 2.0})
    snap.predicted = {"p": 1.0, "T": 305.0}  # T 有变化, p 无变化
    p = snap.to_prompt()
    assert "identifiability=" in p
    assert "σ=" in p  # 只列可辨识 σ
    assert "Δpred=" in p and "Δpred=0" not in p
    assert "'T': 305.0" in p  # 差分只带变化的键
    # 预测与状态一致 → Δpred=0
    snap2 = est.estimate({"p": 1.0, "V": 5.0, "T": 300.0, "n": 2.0})
    snap2.predicted = {"T": 300.0}
    assert "Δpred=0" in snap2.to_prompt()


def test_snapshot_to_prompt_truncates_long_state():
    """超长状态截断: max_state 后加省略号, 避免大状态全刷屏."""
    est = StateEstimator(
        {"space": {"state": ["a", "b", "c", "d", "e", "f", "g", "h"]},
         "observables": ["a"]}
    )
    snap = est.estimate({k: float(i + 1) for i, k in enumerate("abcdefgh")})
    p = snap.to_prompt(max_state=2)
    assert "..." in p
    # 默认 6 键: 8 键仍截断; 显式给足则不截断
    assert len(snap.to_prompt(max_state=20).split("state=")[1]) > len(p.split("state=")[1])


def test_world_catalog_filters_by_domain_on_demand():
    """按需注入: domains={d} 只列该物理域工具, 比全量短 (省 token)."""
    from huginn.security.tool_registry import get_tool

    d = str(get_tool("ideal_gas").schema.get("domain"))
    full = _dummy_mixin()._build_world_catalog_block()
    filtered = _dummy_mixin()._build_world_catalog_block(domains={d})
    assert "已注册解析世界模型" in filtered
    assert "ideal_gas" in filtered
    assert len(filtered.splitlines()) < len(full.splitlines())


def test_matching_domains_drives_on_demand_injection():
    """hypothesis 命中 domain 词 → 只注入该域 (None 否则)."""
    from huginn.security.tool_registry import get_tool

    d = str(get_tool("ideal_gas").schema.get("domain"))
    assert _dummy_mixin()._matching_domains(f"研究 {d} 下的物性") == {d}
    assert _dummy_mixin()._matching_domains("unrelated xyztopic") is None


# ── HarnessDev 结论①: 世界状态持久化 ─────────────────────────
def test_state_snapshot_roundtrip_to_dict():
    """StateSnapshot.to_dict/from_dict 往返无损 (状态/可辨识/后验方差)."""
    from huginn.security.world_state import StateSnapshot

    snap = StateSnapshot(
        state={"p": 1.0, "T": 300.0},
        observables=("p", "T"),
        uncertainty={"p": 0.01, "T": None},
        identifiable_level="limited",
        predicted={"T": 305.0},
        posterior_var={"p": 1e-4, "T": None},
        ts=123456.0,
    )
    back = StateSnapshot.from_dict(snap.to_dict())
    assert back.state == snap.state
    assert back.observables == ("p", "T")
    assert back.uncertainty == {"p": 0.01, "T": None}
    assert back.identifiable_level == "limited"
    assert back.predicted == {"T": 305.0}
    assert back.posterior_var == {"p": 1e-4, "T": None}
    assert back.ts == 123456.0


def test_tracker_save_load_roundtrip(tmp_path):
    """WorldStateTracker.save_state/load_state 跨重启恢复快照与奖励痕迹."""
    from huginn.security.thermo_system import IdealGasWorldModel, ThermoExecutor

    schema = tool_schema("ideal_gas")
    wm = IdealGasWorldModel()
    tracker = WorldStateTracker(schema, wm)
    sb = {"p": 101325.0, "V": 0.022414, "T": 273.15, "n": 1.0}
    tracker.step(sb, PhysicalAction("heat", {"dq": 100.0}))
    exe = ThermoExecutor(initial=dict(sb))
    exe.execute(PhysicalAction("heat", {"dq": 100.0}))
    tracker.observe(exe.observe())
    assert tracker.last_snapshot is not None

    p = str(tmp_path / "ws.json")
    tracker.save_state(p)
    restored = WorldStateTracker.load_state(p)
    assert restored is not None
    assert restored.last_snapshot is not None
    assert restored.last_snapshot.state["p"] == pytest.approx(tracker.last_snapshot.state["p"])
    assert restored.last_prediction_error == pytest.approx(tracker.last_prediction_error)


def test_tracker_load_missing_returns_none(tmp_path):
    assert WorldStateTracker.load_state(str(tmp_path / "nope.json")) is None


# ── 幽灵回流防御 (对齐 Hermes ghost-skill defense) ─────────────
def test_restored_tracker_marked_with_provenance(tmp_path):
    """load_state 恢复的 tracker 带失效标记 (restored/epoch/saved_at), 不冒充当前."""
    from huginn.security.thermo_system import IdealGasWorldModel, ThermoExecutor

    schema = tool_schema("ideal_gas")
    wm = IdealGasWorldModel()
    tracker = WorldStateTracker(schema, wm)
    sb = {"p": 101325.0, "V": 0.022414, "T": 273.15, "n": 1.0}
    tracker.step(sb, PhysicalAction("heat", {"dq": 100.0}))
    exe = ThermoExecutor(initial=dict(sb))
    exe.execute(PhysicalAction("heat", {"dq": 100.0}))
    tracker.observe(exe.observe())
    p = str(tmp_path / "ws.json")
    tracker.save_state(p)
    restored = WorldStateTracker.load_state(p)
    assert restored is not None
    assert restored.is_restored() is True
    assert restored._epoch == tracker._epoch
    assert restored.saved_at is not None


def test_live_tracker_never_stale():
    """未恢复的活运行始终 fresh — 保活尾部, 不误伤当前预测."""
    from huginn.security.thermo_system import IdealGasWorldModel

    tracker = WorldStateTracker(tool_schema("ideal_gas"), IdealGasWorldModel())
    sb = {"p": 101325.0, "V": 0.022414, "T": 273.15, "n": 1.0}
    tracker.step(sb, None)
    assert tracker.is_stale() is False
    assert tracker.fresh_prediction() is tracker.last_prediction  # 未恢复 → 原样返回


def test_restored_stale_blocked_from_feedforward(tmp_path, monkeypatch):
    """失效标记: 写入过久的历史痕迹视为已淘汰, fresh_prediction 拒绝回流."""
    from huginn.security.thermo_system import IdealGasWorldModel, ThermoExecutor

    schema = tool_schema("ideal_gas")
    wm = IdealGasWorldModel()
    tracker = WorldStateTracker(schema, wm)
    sb = {"p": 101325.0, "V": 0.022414, "T": 273.15, "n": 1.0}
    tracker.step(sb, PhysicalAction("heat", {"dq": 100.0}))
    exe = ThermoExecutor(initial=dict(sb))
    exe.execute(PhysicalAction("heat", {"dq": 100.0}))
    tracker.observe(exe.observe())
    p = str(tmp_path / "ws.json")
    tracker.save_state(p)

    restored = WorldStateTracker.load_state(p)
    assert restored.fresh_prediction() == tracker.last_prediction  # 新鲜 → 放行

    # 写入时刻很早 (淘汰) → 应判定失效, 旧预测不得悄悄回流.
    restored.saved_at = 1.0  # 距今远超 _GHOST_MAX_AGE_S
    assert restored.is_stale() is True
    assert restored.fresh_prediction() is None


def test_restored_no_epoch_considered_stale():
    """历史痕迹却无存活观测代 (epoch=0) → 不可信, 按失效处理, 不回流."""
    # 直接构造一个"无存活代"的恢复痕迹 (等价于磁盘里 epoch=0 的旧档).
    tracker = WorldStateTracker.__new__(WorldStateTracker)
    tracker.restored = True
    tracker._epoch = 0
    tracker.saved_at = time.time()
    assert tracker.is_stale() is True
    assert tracker.fresh_prediction() is None
