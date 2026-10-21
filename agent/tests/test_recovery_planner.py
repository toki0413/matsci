"""世界模型故障定位 (P2-1) 红蓝双轨契约门.

蓝队 —— 断言五层各自可达, 且分层顺序符合因果确定性:
    CONTRACT / PRECONDITION / EXECUTION / IDENTIFIABILITY / MODEL_MISMATCH。

红队 —— 主动尝试把"不可归因的偏离"栽给世界模型:
    1) 噪声带内的确认失败不得升为 MODEL_MISMATCH;
    2) 没有预测可比时不得归因为模型失配;
    3) 状态不可辨识时, 残差再大也不得触发"换世界模型/换工具族";
    4) 未知异常保守归执行面, 不得甩锅给模型;
    5) 门自身不得空转 (五层必须都被真实用例命中)。
"""

from __future__ import annotations

import pytest

from huginn.security.recovery_planner import (
    DEFAULT_Z_THRESHOLD,
    FaultLayer,
    FaultObservation,
    locate,
    normalized_residual,
)
from huginn.security.tool_registry import ToolContractError, UnknownToolError
from huginn.security.workspace import DependencyNotMetError, WorkspaceConfirmError
from huginn.security.world_model import ConstraintViolationError, PhysicalAction

ADEQUATE = "adequate"
DEFICIENT = "deficient"


# ── 蓝队: 五层各命中 ───────────────────────────────────────────────
def test_contract_layer_is_not_retried() -> None:
    """契约不符不可恢复 —— 必须落在 CONTRACT 而不是执行面重试."""
    obs = FaultObservation(
        exception=ToolContractError("artifact=2, spec=1"),
        identifiable_level=ADEQUATE,
    )
    assert locate(obs) is FaultLayer.CONTRACT
    assert locate(FaultObservation(exception=UnknownToolError("nope"))) is (
        FaultLayer.CONTRACT
    )


def test_precondition_layer_blocks_before_execution() -> None:
    """约束/依赖不满足是动作未执行 —— 归 PRECONDITION, 与噪声无关."""
    violation = ConstraintViolationError(PhysicalAction("aspirate"), ["源量不足"])
    assert locate(FaultObservation(exception=violation)) is FaultLayer.PRECONDITION
    assert locate(FaultObservation(exception=DependencyNotMetError("s1", "pump"))) is (
        FaultLayer.PRECONDITION
    )


def test_execution_layer_for_process_failures() -> None:
    """进程/IO/超时属执行面故障, 与状态偏离无关, 不受可辨识度影响."""
    assert locate(FaultObservation(exception=TimeoutError("slow"))) is (
        FaultLayer.EXECUTION
    )
    assert locate(FaultObservation(exception=RuntimeError("executor died"))) is (
        FaultLayer.EXECUTION
    )


def test_identifiability_layer_when_state_unobservable() -> None:
    """状态不可辨识时, 再大的残差也只能要求补观测."""
    obs = FaultObservation(
        exception=WorkspaceConfirmError("偏离预期"),
        z_score=100.0,
        identifiable_level=DEFICIENT,
    )
    assert locate(obs) is FaultLayer.IDENTIFIABILITY


def test_model_mismatch_when_deviation_is_significant() -> None:
    """可辨识充分 + 超过 3σ 的确认失败 → 世界模型失配."""
    obs = FaultObservation(
        exception=WorkspaceConfirmError("偏离预期"),
        z_score=DEFAULT_Z_THRESHOLD + 2.0,
        identifiable_level=ADEQUATE,
    )
    assert locate(obs) is FaultLayer.MODEL_MISMATCH


def test_healthy_observation_yields_no_fault() -> None:
    """噪声带内的残差不算故障 —— 不做任何恢复."""
    assert locate(FaultObservation(z_score=2.0, identifiable_level=ADEQUATE)) is None


# ── 红队: 主动尝试错误归因 ────────────────────────────────────────
def test_noise_band_confirm_failure_is_not_model_mismatch() -> None:
    """红队①: 容差比噪声还紧时, 确认失败是检查器问题, 不是模型失配."""
    obs = FaultObservation(
        exception=WorkspaceConfirmError("偏离在噪声带内"),
        z_score=1.2,
        identifiable_level=ADEQUATE,
    )
    layer = locate(obs)
    assert layer is FaultLayer.IDENTIFIABILITY
    assert layer is not FaultLayer.MODEL_MISMATCH


def test_missing_prediction_is_never_blamed_on_model() -> None:
    """红队②: 没有预测就没有"证伪", 不得判模型失配."""
    obs = FaultObservation(
        exception=WorkspaceConfirmError("无预测可对比"),
        z_score=None,
        identifiable_level=ADEQUATE,
    )
    assert locate(obs) is FaultLayer.IDENTIFIABILITY


def test_unidentifiable_state_never_switches_tool_family() -> None:
    """红队③: 不可辨识时残差爆表也不得触发换模型/换工具族."""
    obs = FaultObservation(
        exception=WorkspaceConfirmError("不可辨"),
        z_score=999.0,
        identifiable_level=DEFICIENT,
    )
    assert locate(obs) is not FaultLayer.MODEL_MISMATCH


def test_unknown_exception_stays_at_execution_layer() -> None:
    """红队④: 未知异常保守归执行面, 不甩锅给世界模型."""
    assert locate(FaultObservation(exception=Exception("?"))) is FaultLayer.EXECUTION


def test_gate_is_not_vacuous() -> None:
    """红队⑤: 五层必须都被真实用例命中 —— 门不能形同虚设."""
    cases = [
        FaultObservation(exception=ToolContractError("m"), identifiable_level=ADEQUATE),
        FaultObservation(exception=DependencyNotMetError("s", "d")),
        FaultObservation(exception=RuntimeError("boom")),
        FaultObservation(z_score=50.0, identifiable_level=DEFICIENT),
        FaultObservation(z_score=50.0, identifiable_level=ADEQUATE),
    ]
    hit = {locate(c) for c in cases}
    assert hit == set(FaultLayer)


# ── 归一化残差 ───────────────────────────────────────────────────
def test_normalized_residual_needs_shared_keys() -> None:
    """无共享维度返回 None (不误报零误差)."""
    assert normalized_residual({"a": 1.0}, {"b": 2.0}) is None


def test_normalized_residual_is_scale_free() -> None:
    """σ 归一后不同量级可比: 偏差 = 5σ → z = 5."""
    z = normalized_residual({"p": 1.05}, {"p": 1.00}, sigma={"p": 0.01})
    assert z == pytest.approx(5.0)


def test_normalized_residual_falls_back_on_bad_sigma() -> None:
    """σ 缺失或非正时退回缺省 σ, 不除零、不返回 inf."""
    z = normalized_residual(
        {"p": 0.02}, {"p": 0.0}, sigma={"p": 0.0}, default_sigma=0.01
    )
    assert z == pytest.approx(2.0)


def test_threshold_boundary_is_not_flagged() -> None:
    """恰好等于阈值不算越界 (严格大于才归因)."""
    assert locate(FaultObservation(z_score=DEFAULT_Z_THRESHOLD)) is None
