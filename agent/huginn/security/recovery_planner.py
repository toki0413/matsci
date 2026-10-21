"""世界模型故障定位 —— 故障 = 预测被证伪, 按"在哪一层失配"分层归因.

对标 hexstrike-ai 的错误恢复, 但只借决策形状 (分类 → 候选 → 打分 → 执行),
不借它的不可证伪先验: 它按 stderr 正则分类, ``success_probability`` 是手写常量;
本模块把故障信号收敛到"世界模型预测 vs 实测"的偏离上, 并把**不确定度当作一等的
判据** —— 噪声带内的偏离不得被归因为模型失配 (见红队用例).

五层 (先命中先返回, 不可跨越):

- ``CONTRACT``        : 契约/版本不符. 不可恢复, 换制品而非重试.
- ``PRECONDITION``    : 预演约束 / 前置依赖不满足. 动作未执行, 改动作或改参数.
- ``EXECUTION``       : 进程/IO/超时等执行面故障 (未知异常亦保守归此层) —— 重试/换后端.
- ``IDENTIFIABILITY`` : 可辨识不足, 或偏离落在噪声带内 / 无预测可比 —— 不可归因, 先补观测.
- ``MODEL_MISMATCH``  : 可辨识充分且偏离显著 —— 世界模型该重标定/换代理/换工具族.

本模块**只做定位** (纯函数, 无副作用): 不选恢复策略、不执行动作。策略打分与
``transfer_gate`` 判定属 P2-2, 接线 ``PhysicalWorkspace`` 属 P2-3。
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from huginn.security.tool_registry import ToolContractError, UnknownToolError
from huginn.security.world_model import ConstraintViolationError

__all__ = [
    "DEFAULT_OBS_SIGMA",
    "DEFAULT_Z_THRESHOLD",
    "FaultLayer",
    "FaultObservation",
    "locate",
    "normalized_residual",
]

# 观测噪声缺省 σ (SI 相对量), 与 ``world_state._OBSERVED_SIGMA`` 同义:
# 这里是定位器对"没有标定信息"的兜底, 真实精度应由 sensor_view/标定给出.
DEFAULT_OBS_SIGMA = 0.01

# 3σ 判据: 偏离超过噪声带三倍才允许归因为模型失配.
DEFAULT_Z_THRESHOLD = 3.0

# 可辨识档位命名, 对齐 ``world_state`` 三档 (adequate / limited / deficient).
_DEFICIENT = "deficient"


class FaultLayer(StrEnum):
    """故障归属层 —— 决定后续可选的恢复动作集合 (P2-2 消费)."""

    CONTRACT = "contract"
    PRECONDITION = "precondition"
    EXECUTION = "execution"
    IDENTIFIABILITY = "identifiability"
    MODEL_MISMATCH = "model_mismatch"


@dataclass(frozen=True)
class FaultObservation:
    """一次故障的全部可观测证据 —— 定位器的唯一输入.

    ``identifiable_level`` 缺省为 ``deficient`` (失效安全): 未显式证明可观,
    就不允许把偏差归因到世界模型。
    """

    exception: BaseException | None = None
    # 相对 RMS 预测误差 (与 WorldStateTracker.last_prediction_error 同尺度).
    residual: float | None = None
    # 噪声归一化残差 (|pred-obs|/σ 的 RMS), 量纲无关, 是归因的唯一凭据.
    z_score: float | None = None
    identifiable_level: str = _DEFICIENT

    def has_signal(self, *, z_threshold: float = DEFAULT_Z_THRESHOLD) -> bool:
        """是否有值得定位的故障信号: 有异常, 或残差越过噪声带."""
        if self.exception is not None:
            return True
        return self.z_score is not None and self.z_score > z_threshold


def normalized_residual(
    predicted: Mapping[str, float],
    observed: Mapping[str, float],
    *,
    sigma: Mapping[str, float] | None = None,
    default_sigma: float = DEFAULT_OBS_SIGMA,
) -> float | None:
    """预测 vs 实测的噪声归一化残差 (对角 Mahalanobis 的 RMS 形式).

    逐键 ``z_k = (pred - obs) / σ_k``, 返回 ``sqrt(mean(z_k²))``。
    无共享键 → ``None`` (不误报"零误差": 没有共同维度就谈不上命中)。
    σ 缺失/非正的键退回 ``default_sigma``; 不同量级的状态因此可比。
    """
    shared = [k for k in predicted if k in observed]
    if not shared:
        return None
    sq = 0.0
    for key in shared:
        s = default_sigma
        if sigma is not None:
            given = sigma.get(key)
            if given is not None and float(given) > 0:
                s = float(given)
        sq += ((float(predicted[key]) - float(observed[key])) / s) ** 2
    return math.sqrt(sq / len(shared))


def locate(
    obs: FaultObservation,
    *,
    z_threshold: float = DEFAULT_Z_THRESHOLD,
) -> FaultLayer | None:
    """定位故障层; 无故障信号返回 ``None`` (健康, 不做任何恢复).

    判定顺序即因果确定性顺序: 契约/前置是确定事实, 执行面异常与状态偏离无关,
    最后才用不确定度决定"这次偏离到底能不能算模型的错"。
    """
    if not obs.has_signal(z_threshold=z_threshold):
        return None

    exc = obs.exception

    # L0 契约: 与噪声、模型均无关, 不可恢复 —— 换制品而非重试.
    if isinstance(exc, (ToolContractError, UnknownToolError)):
        return FaultLayer.CONTRACT

    # L1 前置: 动作根本没执行, 因果关系确定.
    if isinstance(exc, ConstraintViolationError) or _is_precondition_error(exc):
        return FaultLayer.PRECONDITION

    # L4 执行面: 进程/IO/超时, 以及未知异常 (保守: 不把未知故障甩给世界模型).
    if exc is not None and not _is_confirm_error(exc):
        return FaultLayer.EXECUTION

    # 到这里只剩"状态偏离"类证据 (感知确认失败, 或纯残差).
    # L2 可辨识: 状态不可观 / 无预测可比 / 偏离落在噪声带内 → 不可归因.
    if obs.identifiable_level == _DEFICIENT:
        return FaultLayer.IDENTIFIABILITY
    if obs.z_score is None or obs.z_score <= z_threshold:
        return FaultLayer.IDENTIFIABILITY

    # L3 模型失配: 可观且显著偏离.
    return FaultLayer.MODEL_MISMATCH


def _is_precondition_error(exc: BaseException | None) -> bool:
    """前置依赖不满足 (延迟导入: workspace 将在 P2-3 反向依赖本模块)."""
    if exc is None:
        return False
    from huginn.security.workspace import DependencyNotMetError

    return isinstance(exc, DependencyNotMetError)


def _is_confirm_error(exc: BaseException | None) -> bool:
    """感知确认失败 —— 动作已执行但状态未达预期 (延迟导入同上)."""
    if exc is None:
        return False
    from huginn.security.workspace import WorkspaceConfirmError

    return isinstance(exc, WorkspaceConfirmError)
