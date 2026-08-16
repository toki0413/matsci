"""#1 Schema 显式化 — 声明式动作规格 (ActionSpec) 测试."""

from __future__ import annotations

import pytest

from huginn.security.physics_schema import (
    PIPETTING_SPEC,
    ActionSpec,
    StepResult,
)
from huginn.security.workspace import (
    DependencyNotMet,
    SimExecutor,
    WorkspaceConfirmError,
)
from huginn.security.world_model import NaiveWorldModel
from huginn.security.experiment_protocol import (
    build_pipette_workflow,
    run_action_spec,
    run_pipette_protocol,
)


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
        id="asp", action_type="aspirate", params={"vol": 10},
        preconditions=["reagent.ready", "pipette.ready"],
        expect=[
            StepResult(key="reagent_vol", expected=88.0, tolerance=0.01),
        ],
    )
    from huginn.security.world_model import PhysicalAction
    with pytest.raises(WorkspaceConfirmError):
        wa.execute(PhysicalAction("aspirate", {"vol": 10}), spec=bad)


def test_spec_preconditions_block_execution():
    """前置依赖不满足 → DependencyNotMet (单步显式调用)."""
    # mixer 缺失 → mixer.ready 不可用
    _, wa = _workflow(mixer_available=False)
    spec = ActionSpec(
        id="mix", action_type="mix", params={"mode": "vortex"},
        preconditions=["mixer.ready", "tube.filled"],
        expect=[],
    )
    from huginn.security.world_model import PhysicalAction
    with pytest.raises(DependencyNotMet):
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
    run_action_spec(wa_spec)          # spec 驱动 (sim=True)
    run_pipette_protocol(wa_hard)     # 硬编码
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