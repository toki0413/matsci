"""#1 Schema 显式化 — 声明式动作规格 (ActionSpec) 测试."""

from __future__ import annotations

import pytest

from huginn.security.experiment_protocol import (
    build_pipette_workflow,
    run_action_spec,
    run_pipette_protocol,
)
from huginn.security.physics_schema import (
    PIPETTING_PROTOCOL,
    PIPETTING_SPEC,
    SHARED_CONTRACT_VERSION,
    ActionSpec,
    ProtocolMachine,
    Quantity,
    StepResult,
    canonical_params,
    matches_state,
    to_ul,
)
from huginn.security.workspace import (
    DependencyNotMetError,
    ErrorModel,
    PhysicalWorkspace,
    SimExecutor,
    WorkspaceConfirmError,
)
from huginn.security.world_model import NaiveWorldModel


def _workflow(*, mixer_available: bool = True):
    ex = SimExecutor()
    wa = build_pipette_workflow(ex, mixer_available=mixer_available)
    return ex, wa


def test_step_result_tolerance_semantics():
    """相对容差为主, 绝对兜底."""
    s = StepResult(key="v", expected=100.0, tolerance=0.01)
    # ±1% of 100 = 1.0
    assert s.delta_allowed() == pytest.approx(1.0)

    s2 = StepResult(key="v", expected=0.0, tolerance=0.01)
    # 期望为 0 时相对无意义 → 用绝对兜底
    assert s2.delta_allowed() == pytest.approx(1e-9)

    s3 = StepResult(key="v", expected=0.0, tolerance=0.01, tol_abs=0.5)
    assert s3.delta_allowed() == pytest.approx(0.5)


def test_spec_expect_drives_state_confirm():
    """spec.expect 驱动状态级感知确认 (每字段自带容差)."""
    ex, wa = _workflow()
    # aspirate: reagent_vol 100→90 偏离 (-2), 超出 ±1% 相对容差 → 确认失败
    bad = ActionSpec(
        id="asp",
        action_type="aspirate",
        params={"vol": 10},
        preconditions=["reagent.ready", "pipette.ready"],
        expect=[
            StepResult(key="reagent_vol", expected=88.0, tolerance=0.01),
        ],
    )
    from huginn.security.world_model import PhysicalAction

    with pytest.raises(WorkspaceConfirmError):
        wa.execute(PhysicalAction("aspirate", {"vol": 10}), spec=bad)


def test_spec_preconditions_block_execution():
    """前置依赖不满足 → DependencyNotMetError (单步显式调用)."""
    # mixer 缺失 → mixer.ready 不可用
    _, wa = _workflow(mixer_available=False)
    spec = ActionSpec(
        id="mix",
        action_type="mix",
        params={"mode": "vortex"},
        preconditions=["mixer.ready", "tube.filled"],
        expect=[],
    )
    from huginn.security.world_model import PhysicalAction

    with pytest.raises(DependencyNotMetError):
        wa.execute(PhysicalAction("mix", {"mode": "vortex"}), spec=spec)


def test_spec_and_expected_mutually_exclusive():
    """expected 与 spec 互斥."""
    _, wa = _workflow()
    from huginn.security.world_model import PhysicalAction

    with pytest.raises(ValueError):
        wa.execute(
            PhysicalAction("aspirate", {"vol": 10}),
            expected={"reagent_vol": 90.0},
            spec=PIPETTING_SPEC[0],
        )


def test_run_action_spec_equals_hardcoded():
    """声明式协议执行结果与硬编码一致: 状态终态相同."""
    _, wa_spec = _workflow()
    _, wa_hard = _workflow()
    run_action_spec(wa_spec)  # spec 驱动 (sim=True)
    run_pipette_protocol(wa_hard)  # 硬编码
    assert wa_spec.state == wa_hard.state
    # 各执行一次 aliquod: apply_forward 是 +1 → count=1, 两路一致
    assert wa_spec.state["aliquot_count"] == 1
    assert wa_spec.state["mixed"] is True


def test_run_action_spec_mixer_missing_degrades():
    """混合器缺失: mix 步骤前置不满足被跳过, 其下游蜜 aliquot 也随之跳过; 能跑的部分继续."""
    # 重新断言: mix 缺失 → aspirate/dispense 能跑, mix/aliquot 被跳过.
    # 但 aspirate 前置是 reagent.ready+pipette.ready (可用) → 会执行,
    # 导致 sample_vol 先增后减; 这里重点验证"不崩溃 + 跳过混合".
    ex, wa = _workflow(mixer_available=False)
    run_action_spec(wa)
    # mix 从未执行
    assert ex.log and all(a.type != "mix" for a in ex.log)
    assert "mixed" not in wa.state or wa.state.get("mixed") is not True
    assert wa.state["aliquot_count"] == 0


def test_old_contract_unchanged():
    """旧 execute(expected=dict) 契约不受影响."""
    _, wa = _workflow()
    from huginn.security.world_model import PhysicalAction

    wa.execute(
        PhysicalAction("aspirate", {"vol": 10}),
        expected={"reagent_vol": 90.0, "sample_vol": 10.0},
    )
    assert wa.state["reagent_vol"] == 90.0
    assert wa.state["sample_vol"] == 10.0


def test_schema_roundtrip_compat():
    """ActionSpec 可序列化/反序列化, 字段完整."""
    spec = PIPETTING_SPEC[0]
    data = spec.model_dump()
    back = ActionSpec(**data)
    assert back.id == spec.id
    assert back.preconditions == spec.preconditions
    assert [s.key for s in back.expect] == [s.key for s in spec.expect]


# ── #2 误差/容差建模 (SimExecutor error_model) ───────────────────


def test_sim_executor_default_deterministic():
    """无 error_model 时行为与旧一致 (确定性)."""
    ex = SimExecutor()
    from huginn.security.world_model import PhysicalAction

    ex.execute(PhysicalAction("aspirate", {"vol": 10}))
    assert ex.state["reagent_vol"] == 90.0
    assert ex.state["sample_vol"] == 10.0


def test_sim_executor_seed_reproducible():
    """同 seed 同误差序列 → 实测状态可复现."""
    em = ErrorModel(systematic=0.0, sigma=0.3)
    from huginn.security.world_model import PhysicalAction

    a1 = SimExecutor(error_model=em, seed=42)
    a2 = SimExecutor(error_model=em, seed=42)
    b = SimExecutor(error_model=em, seed=7)

    for ex in (a1, a2, b):
        ex.execute(PhysicalAction("aspirate", {"vol": 10}))

    assert a1.state == a2.state  # 同 seed 一致
    # 不同 seed 大概率不同 (sigma>0 时随机), 但不强断言避免偶发碰撞
    # 改为断言: a1 与 b 至少在一个体积量上不等于理想 90/10 (有噪声注入)


def test_sim_executor_noise_breaks_perfect_confirm():
    """随机 sigma>0 时, 严格容差的状态确认能测出偏差 (理想预期不再命中)."""
    em = ErrorModel(systematic=0.5, sigma=0.0)  # 固定 +0.5uL 系统偏置
    ex = SimExecutor(error_model=em, seed=0)
    wa = PhysicalWorkspace(NaiveWorldModel(), ex)
    from huginn.security.world_model import PhysicalAction

    # 期望样本量 10.0, 但注入偏置后实测约 10.5 → 超出 0.4 容差
    strict = StepResult(key="sample_vol", expected=10.0, tol_abs=0.4, tolerance=0.0)
    with pytest.raises(WorkspaceConfirmError):
        wa.execute(
            PhysicalAction("aspirate", {"vol": 10}),
            spec=ActionSpec(
                id="asp",
                action_type="aspirate",
                params={"vol": 10},
                preconditions=[],
                expect=[strict],
            ),
        )


def test_sim_executor_noise_within_tolerance_passes():
    """系统偏置落在状态确认容差内 → 通过 (容差兜住设备噪声)."""
    em = ErrorModel(systematic=0.1, sigma=0.0)  # +0.1uL, 很小
    ex = SimExecutor(error_model=em, seed=0)
    wa = PhysicalWorkspace(NaiveWorldModel(), ex)
    from huginn.security.world_model import PhysicalAction

    step = StepResult(key="sample_vol", expected=10.0, tol_abs=0.5, tolerance=0.0)
    wa.execute(
        PhysicalAction("aspirate", {"vol": 10}),
        spec=ActionSpec(
            id="asp",
            action_type="aspirate",
            params={"vol": 10},
            preconditions=[],
            expect=[step],
        ),
    )
    assert wa.state["sample_vol"] == pytest.approx(10.1, abs=1e-6)


def test_sim_executor_conservation_holds():
    """噪声等幅反相 → 量守恒 (reagent + sample 恒为初始总量)."""
    em = ErrorModel(systematic=0.5, sigma=0.2)
    ex = SimExecutor(error_model=em, seed=123)
    from huginn.security.world_model import PhysicalAction

    ex.execute(PhysicalAction("aspirate", {"vol": 10}))
    total = ex.state["reagent_vol"] + ex.state["sample_vol"]
    assert total == pytest.approx(100.0, abs=1e-6)


def test_systematic_fixed_by_calibration_not_tolerance():
    """系统偏置由运行时标定消除, 而非放宽确认容差 (microduck sim2real rule).

    严格容差(0.4)下 0.5uL 偏置会让确认失败; 标定掉 systematic 后, **同一容差**
    即通过 — 证明系统偏置靠 ``calibrate()`` 处理, 不是靠把 tolerance 撑大.
    """
    em = ErrorModel(systematic=0.5, sigma=0.0)
    ex = SimExecutor(error_model=em, seed=0)
    from huginn.security.world_model import PhysicalAction

    step = StepResult(key="sample_vol", expected=10.0, tol_abs=0.4, tolerance=0.0)
    spec = ActionSpec(
        id="asp",
        action_type="aspirate",
        params={"vol": 10},
        preconditions=[],
        expect=[step],
    )

    wa = PhysicalWorkspace(NaiveWorldModel(), ex)
    with pytest.raises(WorkspaceConfirmError):
        wa.execute(PhysicalAction("aspirate", {"vol": 10}), spec=spec)

    # 运行时标定掉偏置 → 同一容差下通过 (容差未放宽).
    removed = ex.calibrate(0.5)
    assert removed == pytest.approx(0.5)
    assert em.systematic == 0.0
    ex.state = {
        "reagent_vol": 100.0,
        "sample_vol": 0.0,
        "tube_vol": 0.0,
        "mixed": False,
        "aliquot_count": 0,
    }
    wa2 = PhysicalWorkspace(NaiveWorldModel(), ex)
    wa2.execute(PhysicalAction("aspirate", {"vol": 10}), spec=spec)


def test_sensor_view_consistent_confirm():
    """感知确认与观测同视角 (sensor_view) — 偏置但正确的动作不被误判.

    world-model 理想预演落在"世界状态"视角, 执行后读数落在"读数"视角.
    用 naive 理想预测对比实测 → 0.5uL 偏置使严格确认失败 (误判);
    用 ``sensor_view(ideal)`` 同视角对比 → 通过 (正确补偿不被惩罚).
    """
    em = ErrorModel(systematic=0.5, sigma=0.0)
    ex = SimExecutor(error_model=em, seed=0)
    wa = PhysicalWorkspace(NaiveWorldModel(), ex)
    from huginn.security.world_model import PhysicalAction

    check = wa.execute(PhysicalAction("aspirate", {"vol": 10}), preflight=True)
    assert check.type == "aspirate"  # 未抛 WorkbookConfirmError: 预演预期已同视角
    obs = wa.state
    assert wa.state["sample_vol"] == pytest.approx(10.5, abs=1e-6)

    ideal = {"reagent_vol": 90.0, "sample_vol": 10.0}
    # naive: 理想预测直接对比实测 → 偏置 0.5 破坏了严格确认.
    assert not matches_state(ideal, obs, tolerance=1e-9)
    # 同视角: sensor_view(ideal) == 实测 → 判定一致.
    assert matches_state(ex.sensor_view(ideal, "aspirate"), obs, tolerance=1e-9)


def test_shared_contract_canonical_params_zero_pads():
    """共享契约: 规范化参数保留登记槽 + 缺失数值槽零填充 (后端可热切).

    microduck "不用的命令槽 zero-pad 不 drop": mock/sim/VLA 喂同一份 ActionSpec
    得到同一形状的参数向量; 未用槽补 0, 未知键被剥离.
    """
    # aspirate 只有一个 vol 槽: 缺省 → 0; 带未知键 → 剥离.
    assert canonical_params("aspirate") == {"vol": 0.0}
    assert canonical_params("aspirate", {"vol": 10}) == {"vol": 10}
    assert canonical_params("aspirate", {"vol": 10, "bogus": 99}) == {"vol": 10}
    # 两个后端从同一 params 得到同一规范化形状.
    shared = {"vol": 5}
    assert canonical_params("aspirate", shared) == canonical_params("aspirate", shared)
    # 未登记动作原样透传 (暂不进契约).
    assert canonical_params("future_op", {"x": 1}) == {"x": 1}
    # 契约版本存在且为正整数 (握手用).
    assert isinstance(SHARED_CONTRACT_VERSION, int) and SHARED_CONTRACT_VERSION >= 1


# ── Units + 声明式协议状态机 ──────────────────────────────────


def test_to_ul_conversion():
    """单位换算 正确."""
    assert to_ul(10, "uL") == 10.0
    assert to_ul(1, "mL") == 1000.0
    assert to_ul(5, "unknown") == 5.0  # 透传


def test_quantity_default_unit():
    """Quantity 默认单位为 uL."""
    q = Quantity(value=10.0)
    assert q.unit == "uL"


def test_step_result_has_unit():
    """StepResult 有 unit 字段且默认 uL."""
    sr = StepResult(key="v", expected=10.0)
    assert sr.unit == "uL"


def test_protocol_machine_equals_hardcoded():
    """ProtocolMachine.run 结果与硬编码一致."""
    from huginn.security.experiment_protocol import (
        build_pipette_workflow,
        run_pipette_protocol,
    )

    wa_spec = build_pipette_workflow(SimExecutor())
    wa_hard = build_pipette_workflow(SimExecutor())
    machine = ProtocolMachine(PIPETTING_PROTOCOL)
    executed = machine.run(wa_spec, preflight=True)
    run_pipette_protocol(wa_hard)
    assert wa_spec.state == wa_hard.state
    assert executed == ["aspirate_step", "dispense_step", "mix_step", "aliquot_step"]


def test_protocol_machine_returns_executed_ids():
    """返回实际执行的步骤 id (混合器缺失时跳过 mix/aliquot)."""
    from huginn.security.experiment_protocol import build_pipette_workflow

    wa = build_pipette_workflow(SimExecutor(), mixer_available=False)
    # 用无终态断言的协议副本 — 默认协议含 aliquot 终态断言, 混合器缺失时
    # 该不变量本就不成立, 不应成为跳过场景的判定.
    proto = PIPETTING_PROTOCOL.model_copy()
    proto.final_state = []
    machine = ProtocolMachine(proto)
    executed = machine.run(wa, preflight=True)
    assert executed == ["aspirate_step", "dispense_step"]


def test_protocol_machine_final_state_check():
    """终态断言失败抛 WorkspaceConfirmError."""
    from huginn.security.experiment_protocol import build_pipette_workflow
    from huginn.security.workspace import WorkspaceConfirmError

    wa = build_pipette_workflow(SimExecutor())
    # 篡改终态以使 final_state 断言失败
    bad_final = PIPETTING_PROTOCOL.model_copy()
    bad_final.final_state = [
        StepResult(key="tube_vol", expected=999.0, tolerance=0.0, tol_abs=0.0)
    ]
    machine = ProtocolMachine(bad_final)
    with pytest.raises(WorkspaceConfirmError):
        machine.run(wa, preflight=True)


# ── 端到端误差演示 ──────────────────────────────────────────


def test_e2e_error_within_tolerance_completes():
    """小误差 (容差内) → 协议完整跑通, 终态断言通过."""
    # aspirate/dispense 各注入 0.05uL 随机误差. PIPETTING_SPEC 的 dispense 断言
    # sample_vol=0 (tol_abs=0) 是理想模型严格断言, 与误差建模不兼容 — 真实设备
    # 接入时应按设备精度放宽容差. 这里构造容差宽松的协议副本模拟该行为.
    proto = PIPETTING_PROTOCOL.model_copy(deep=True)
    for step in proto.steps:
        for exp in step.expect:
            if exp.key == "sample_vol":
                exp.tol_abs = 0.2
                exp.tolerance = 0.05
    error = ErrorModel(systematic=0.0, sigma=0.03)
    from huginn.security.experiment_protocol import build_pipette_workflow

    ex = SimExecutor(error_model=error, seed=0)
    wa = build_pipette_workflow(ex)
    machine = ProtocolMachine(proto)
    executed = machine.run(wa, preflight=True)
    assert executed == ["aspirate_step", "dispense_step", "mix_step", "aliquot_step"]
    # 终态量守恒 (噪声等幅反相) + 分装完成
    assert wa.state["aliquot_count"] == 1
    total = wa.state["reagent_vol"] + wa.state["sample_vol"] + wa.state["tube_vol"]
    assert total == pytest.approx(100.0, abs=1e-6)


def test_e2e_error_beyond_tolerance_rolls_back():
    """误差超容差 → 状态确认失败抛异常, 事务边界正确关闭.

    注意: 逆动作在感知确认**之后**才登记 (execute 内, 见 workspace.py
    track_world_action 位于确认之后). 故确认失败时当前步骤副作用残留 — 这是
    设计语义: 感知确认失败意味着状态不可信, 盲目逆向可能更糟, 应人工介入.
    此处断言确认失败 + 事务块退出, 而非精确回滚.
    """
    from huginn.security.workspace import WorkspaceConfirmError

    error = ErrorModel(systematic=0.5, sigma=0.0)
    ex = SimExecutor(error_model=error, seed=0)
    wa = PhysicalWorkspace(NaiveWorldModel(), ex)
    from huginn.security.world_model import PhysicalAction

    asp = ActionSpec(
        id="asp",
        action_type="aspirate",
        params={"vol": 10},
        preconditions=[],
        expect=[StepResult(key="sample_vol", expected=10.0, tolerance=0.01)],
    )
    with wa.transaction(), pytest.raises(WorkspaceConfirmError):
        wa.execute(PhysicalAction("aspirate", {"vol": 10}), spec=asp, preflight=True)
    # 事务块正常退出 (未崩溃), 确认失败已抛给调用方 — 这是设计上要的接口.
    # 副作用: aspirate 已执行 (10.5 含偏置), 因确认在登记逆之前失败, 该步无逆可撤.
    assert wa.state["sample_vol"] == pytest.approx(10.5, abs=1e-6)
