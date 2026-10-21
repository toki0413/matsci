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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from huginn.security.tool_registry import (
    ToolContractError,
    UnknownToolError,
    borrow_verdict,
)
from huginn.security.world_model import ConstraintViolationError

__all__ = [
    "DEFAULT_OBS_SIGMA",
    "DEFAULT_Z_THRESHOLD",
    "FaultLayer",
    "FaultObservation",
    "Observation",
    "Provenance",
    "RecoveryAction",
    "RecoveryStep",
    "UnmeasuredObservationError",
    "degraded_predict",
    "locate",
    "normalized_residual",
    "plan",
    "recovery_block",
    "require_measured",
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


# ── P2-2: 层内候选 + 下界打分 (只出建议, 不执行) ─────────────────────
class RecoveryAction(StrEnum):
    """恢复动作. 层是不可跨越的硬门, 打分只在门内排序."""

    REPLACE_ARTIFACT = "replace_artifact"
    AMEND_ACTION = "amend_action"
    OBSERVE = "observe"
    RETRY = "retry"
    SWITCH_BACKEND = "switch_backend"
    SWITCH_TOOL = "switch_tool"
    RECALIBRATE = "recalibrate"
    SWITCH_WORLD_MODEL = "switch_world_model"
    DEGRADED_PROXY = "degraded_proxy"
    ROLLBACK = "rollback"
    ESCALATE = "escalate"


_LAYER_ACTIONS: dict[FaultLayer, tuple[RecoveryAction, ...]] = {
    FaultLayer.CONTRACT: (RecoveryAction.REPLACE_ARTIFACT,),
    FaultLayer.PRECONDITION: (RecoveryAction.AMEND_ACTION,),
    FaultLayer.IDENTIFIABILITY: (RecoveryAction.OBSERVE,),
    FaultLayer.MODEL_MISMATCH: (
        RecoveryAction.RECALIBRATE,
        RecoveryAction.SWITCH_WORLD_MODEL,
        RecoveryAction.SWITCH_TOOL,
    ),
    FaultLayer.EXECUTION: (
        RecoveryAction.RETRY,
        RecoveryAction.SWITCH_BACKEND,
        RecoveryAction.DEGRADED_PROXY,
        RecoveryAction.ROLLBACK,
    ),
}

# 会再次对物理世界施加动作的策略 —— 不可逆现场一律不得自动执行.
_MUTATING = frozenset(
    {
        RecoveryAction.RETRY,
        RecoveryAction.SWITCH_BACKEND,
        RecoveryAction.SWITCH_TOOL,
        RecoveryAction.ROLLBACK,
    }
)

# 每 1s 实测耗时折价 0.001 后验下界 (100s ≈ 降 0.1); 下界 z=1 ≈ 84% 置信.
_COST_WEIGHT = 0.001
_LCB_Z = 1.0


@dataclass(frozen=True)
class RecoveryStep:
    """一条排序后的恢复建议 (advisory, 由上层决定是否采纳)."""

    action: RecoveryAction
    label: str
    score: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "label": self.label,
            "score": round(self.score, 4),
            "reason": self.reason,
        }


def _beta_lcb(successes: int, failures: int, z: float = _LCB_Z) -> float:
    """Beta(1+s, 1+f) 的下界 —— 用分位数而非均值, 避免偏好"高收益偶尔毁现场"."""
    a = 1.0 + max(successes, 0)
    b = 1.0 + max(failures, 0)
    mean = a / (a + b)
    var = (a * b) / ((a + b) ** 2 * (a + b + 1.0))
    return max(0.0, min(1.0, mean - z * math.sqrt(var)))


def _posterior_counts(
    bandit: Any | None,
    objective: str,
    action: RecoveryAction,
    label: str,
) -> tuple[int, int]:
    """从 WorkflowBandit 取 (successes, failures); 取不到按 0/0 (先验 Beta(1,1))."""
    if bandit is None or not objective:
        return 0, 0
    try:
        belief = bandit.get_belief(f"{action.value}:{label}", objective)
    except Exception:  # 防御: 后续打分失败退回冷启动先验
        return 0, 0
    if belief is None:
        return 0, 0
    return int(getattr(belief, "successes", 0)), int(getattr(belief, "failures", 0))


def plan(
    obs: FaultObservation,
    *,
    tool: str | None = None,
    alternatives: Sequence[str] = (),
    bandit: Any | None = None,
    costs: Mapping[str, float] | None = None,
    reversible: bool = True,
    retry_index: int = 0,
    max_retries: int = 2,
    objective: str = "",
) -> list[RecoveryStep]:
    """定位 → 硬门筛选 → 打分排序, 返回全部候选 (首个即推荐).

    硬门 (不参与打分, 越不过去): 层内动作集 / ``transfer_gate`` 转用判定 /
    不可逆禁自动执行 / 重试预算。打分只做门内排序, 先验来自 bandit 后验,
    成本来自实测耗时 ``costs`` (键 = 工具名或动作名)。
    """
    layer = locate(obs)
    if layer is None:
        return []

    allowed = list(_LAYER_ACTIONS[layer])
    gated: list[tuple[str, str]] = []
    if RecoveryAction.SWITCH_TOOL in allowed:
        if tool is None:
            allowed.remove(RecoveryAction.SWITCH_TOOL)
        else:
            for alt in alternatives:
                if alt == tool:
                    continue
                try:
                    verdict = borrow_verdict(tool, alt)
                except Exception:  # 防御: 判定不可用按隔离失效安全跳过
                    continue
                if verdict.allow_machinery:
                    gated.append((alt, verdict.reason))
            if not gated:
                allowed.remove(RecoveryAction.SWITCH_TOOL)

    if not reversible:
        allowed = [a for a in allowed if a not in _MUTATING]
    if retry_index >= max_retries and RecoveryAction.RETRY in allowed:
        allowed.remove(RecoveryAction.RETRY)

    steps: list[RecoveryStep] = []
    for action in allowed:
        labels = (
            [alt for alt, _ in gated] if action is RecoveryAction.SWITCH_TOOL else [""]
        )
        for label in labels:
            counts = _posterior_counts(bandit, objective, action, label)
            cost = float((costs or {}).get(label or action.value, 0.0))
            steps.append(
                RecoveryStep(
                    action=action,
                    label=label,
                    score=_beta_lcb(*counts) - _COST_WEIGHT * cost,
                    reason=_reason(action, label, gated),
                )
            )

    if not steps:
        steps.append(
            RecoveryStep(RecoveryAction.ESCALATE, "", 0.0, "无安全候选, 交人工")
        )
    steps.sort(key=lambda s: (-s.score, s.action.value, s.label))
    return steps


_REASONS: dict[RecoveryAction, str] = {
    RecoveryAction.REPLACE_ARTIFACT: "契约/版本不符, 重试无意义",
    RecoveryAction.AMEND_ACTION: "前置不满足, 动作未执行; 改参数或改动作后重新预演",
    RecoveryAction.OBSERVE: "证据不足或落在噪声带内, 先补观测再归因",
    RecoveryAction.RETRY: "执行面瞬时故障, 预算内重试",
    RecoveryAction.SWITCH_BACKEND: "换执行后端 (仿真/真实/备用)",
    RecoveryAction.RECALIBRATE: "用残差回流重标定代理/噪声模型",
    RecoveryAction.SWITCH_WORLD_MODEL: "解析模型与实测系统性不符, 换代理模型",
    RecoveryAction.DEGRADED_PROXY: "真工具不可用, 用代理预测并显式标注",
    RecoveryAction.ROLLBACK: "撤销本步副作用, 回到执行前状态",
    RecoveryAction.ESCALATE: "无安全候选, 交人工",
}


def _reason(
    action: RecoveryAction, label: str, gated: Sequence[tuple[str, str]]
) -> str:
    if action is RecoveryAction.SWITCH_TOOL:
        for alt, why in gated:
            if alt == label:
                return f"{why}; 换入后须按本域重标定"
    return _REASONS[action]


def recovery_block(
    layer: FaultLayer | None,
    steps: Sequence[RecoveryStep],
    *,
    max_items: int = 3,
) -> str:
    """把建议格式化成可注入 prompt 的 ``[RECOVERY]`` 块 (advisory)."""
    if layer is None or not steps:
        return ""
    lines = [f"[RECOVERY] 故障层: {layer.value} (建议, 不自动执行):"]
    for i, step in enumerate(steps[:max_items], 1):
        label = f" [{step.label}]" if step.label else ""
        lines.append(f"  {i}. {step.action.value}{label} — {step.reason}")
    return "\n".join(lines) + "\n"


# ── P2-4: 代理降级 —— 预测永不冒充实测 ─────────────────────────────
class Provenance(StrEnum):
    MEASURED = "measured"
    PREDICTED = "predicted"


class UnmeasuredObservationError(RuntimeError):
    """试图把代理预测当作实测写入状态/记忆."""


@dataclass(frozen=True)
class Observation:
    """带来源标记的观测. ``PREDICTED`` 不得进入状态估计或长期记忆."""

    values: Mapping[str, float]
    provenance: Provenance
    source: str
    degraded: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "values": dict(self.values),
            "provenance": self.provenance.value,
            "source": self.source,
            "degraded": self.degraded,
        }


def require_measured(obs: Observation) -> dict[str, float]:
    """写入状态/记忆前的唯一闸门: 非实测一律拒绝."""
    if obs.provenance is not Provenance.MEASURED:
        raise UnmeasuredObservationError(
            f"拒绝写入非实测观测: provenance={obs.provenance.value}, source={obs.source}"
        )
    return dict(obs.values)


def degraded_predict(
    predictor: Any,
    state: Mapping[str, Any],
    action: Any,
    *,
    reason: str = "backend unavailable",
) -> Observation | None:
    """真工具不可用时的代理预测; 无预测能力/输出非法 → ``None`` (不臆造)."""
    try:
        predicted = predictor.predict(state, action)
    except Exception:  # 防御: 代理不可用即降级失败, 不伪造
        return None
    if not predicted:
        return None
    try:
        values = {str(k): float(v) for k, v in predicted.items()}
    except (TypeError, ValueError):
        return None
    return Observation(
        values=values,
        provenance=Provenance.PREDICTED,
        source=f"surrogate:{type(predictor).__name__} ({reason})",
        degraded=True,
    )
