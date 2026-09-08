"""世界模型核心 —— 对标 VLA / 机器人模型驱动规划, 以**数学语言**为规划与验证的载体.

把「搜读算做写」里的科研决策, 从散文式 prompt 升级为「状态/动作/转移(定律)」的
形式化对象, 使规划与验证都能用**方程与数值对账**表达 (数学即核心):

  - :class:`WorldState`  世界可测状态(科学系统的数值特征向量, 如 a_AU/S/T_eq)。
  - :class:`Action`      可执行动作(实验/仿真配置, 如轨道半长轴缩放 a_scale)。
  - :class:`WorldModel`  世界的转移模型 —— ``predict(state, action) -> state'``,
    其 ``law()`` 返回该域的**数学定律(方程串)**。这是"模型基规划"里的 forward model。
  - :func:`reconcile`    预测 vs 真实执行的数值对账: 判定定律是否被证实/证伪
    (不悄悄覆盖 —— 偏差本身就是科学发现)。
  - :class:`ModelBasedPlanner` 在状态空间里 rollout: 对每个候选动作用世界模型预告后继
    状态, 按预测目标排序产出**计划即数学** (每个 PlanStep = 动作 + 定律 + 预测状态)。

设计对照:
  - 机器人分层规划/世界模型: 高层 goal →(本次) 状态空间 rollout 预告, 再落地真实执行。
  - VLA: 感知(读入状态) → 语言+定律推理(这里是方程) → 动作 → 验证(对账) → 微调。
  - 数学核心: law() 是符号方程, predict() 是它的数值实现, reconcile() 是它的可证伪性
    校验 —— 全程无语义幻觉: 一个数值来自哪个定律、预测对否, 都可审计。

诚实边界: 世界模型是**假说**, execute 是**真相检验**。predict 只做预告, 不替代执行;
reconcile 把预测与真值并排审计, 不符则如实标 falsified。
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

# 基础物理常数 (世界模型的数学参数) —— SI 单位
SOLAR_LUM = 3.828e26          # L☉ [W]
STEFAN = 5.670374419e-8        # σ [W/m²/K⁴]
AU_M = 1.49597870700e11        # [m]
SOLAR_MASS_KG = 1.98892e30


def kepler_semimajor_au(period_d: float, host_msun: float = 1.0) -> float:
    """开普勒第三定律: a[AU] = (M*[M☉])^{1/3} (P[yr])^{2/3}. (数学定律)"""
    return host_msun ** (1 / 3) * (period_d / 365.25) ** (2 / 3)


# ── 状态 / 动作 (形式化的世界) ──────────────────────────────────


@dataclass
class WorldState:
    """世界可测状态: 数值特征向量 (如 a_AU / S_Wm2 / T_eq_K)."""
    vector: dict[str, float] = field(default_factory=dict)
    domain: str = ""

    def __getitem__(self, k: str) -> float:
        return self.vector[k]

    def get(self, k: str, default: float = 0.0) -> float:
        return self.vector.get(k, default)

    def as_dict(self) -> dict:
        return {"domain": self.domain, "state": dict(self.vector)}


@dataclass
class Action:
    """可执行动作: 实验/仿真配置 (参数向量)."""
    config: dict[str, float] = field(default_factory=dict)
    label: str = ""

    def as_dict(self) -> dict:
        return {"label": self.label, "action": dict(self.config)}


@dataclass
class PlanStep:
    """计划即数学: 一个动作 + 其数学定律 + 世界模型预告的下继状态."""
    action: Action
    law: str = ""                        # 数学语言核心: 该步依据的方程
    predicted: WorldState | None = None  # predict(state, action)

    def to_dict(self) -> dict:
        return {
            "action": self.action.as_dict(),
            "law": self.law,
            "predicted": self.predicted.as_dict() if self.predicted else None,
        }


# ── 世界模型 (forward model) ────────────────────────────────────


class WorldModel(ABC):
    """世界的转移模型: predict(state, action) -> next state, law() 给数学方程. """

    domain = ""

    @abstractmethod
    def law(self) -> str:
        """返回该世界的数学定律(方程串)—— 规划/预告/验证共享的证据来源."""

    @abstractmethod
    def predict(self, state: WorldState, action: Action) -> WorldState:
        """预告: 施加动作后世界应转移到的状态 (数学定律的数值实现)."""

    @abstractmethod
    def seed(self, observation: dict[str, Any]) -> WorldState:
        """从外部观测(读入的科学数据)落下初始状态."""


class FirstPrinciplesWorldModel(WorldModel):
    """系外行星第一性原理世界模型: 轨道日晒 - 黑体平衡温度.

    law (数学):

        a = (M*/M☉)^{1/3} (P/yr)^{2/3} [AU]           开普勒第三定律
        S = L☉ / (4π a²)  [W/m²]                      日晒(各向同性辐射)
        T_eq = ((1-α) S / (4σ))^{1/4}  [K]             黑体平衡温度(Stefan-Boltzmann)

    动作空间: Action.config = {"a_scale": s}   → 半长轴缩放 a' = a·s (扰动轨道假设).
    """

    domain = "exoplanet"

    def __init__(self, albedo: float = 0.1, host_msun: float = 1.0) -> None:
        self.albedo = albedo
        self.host_msun = host_msun

    def law(self) -> str:
        return ("T_eq = ((1-α)·S/(4σ))^{1/4};  S = L☉/(4πa²);  "
                f"a = ({self.host_msun} M☉)^{1/3}(P/yr)^{{2/3}} AU;  α={self.albedo}")

    def seed(self, observation: dict[str, Any]) -> WorldState:
        p_d = float(observation["orbper_d"])
        a_au = kepler_semimajor_au(p_d, self.host_msun)
        return WorldState({"P_d": p_d, "a_AU": a_au}, domain=self.domain)

    def predict(self, state: WorldState, action: Action) -> WorldState:
        a_au = state.get("a_AU", 1.0) * action.config.get("a_scale", 1.0)
        a_m = a_au * AU_M
        S = SOLAR_LUM / (4 * math.pi * a_m ** 2)
        T_eq = ((1 - self.albedo) * S / (4 * STEFAN)) ** 0.25
        return WorldState({"a_AU": round(a_au, 4),
                           "S_Wm2": round(S, 1),
                           "T_eq_K": round(T_eq, 1)}, domain=self.domain)


# ── 预测 vs 真实 的数学对账 (可证伪性校验) ──────────────────────


def reconcile(predicted: WorldState, actual: dict, *, tol: float = 0.03,
              metrics: tuple[str, ...] = ("S_Wm2", "T_eq_K")) -> dict:
    """预测 vs 真实执行的数值对账.

    每个可测指标算相对误差; 全部 ≤ tol → 定律被当前证据**证实**(borne_out=True),
    否则如实标 **falsified**(定律是假说, 偏差不覆盖 —— 本身就是发现)。

    Returns: {"borne_out": bool, "errors": {metric: rel_err}, "mismatch": {...}}
    """
    errors: dict[str, float] = {}
    mismatch: dict[str, tuple[float, float]] = {}
    for m in metrics:
        y_m = predicted.get(m)
        y_e = float(actual.get(m, float("nan")))
        if y_m is None or math.isnan(y_e):
            continue
        err = abs(y_m - y_e) / (abs(y_e) if abs(y_e) > 1e-12 else 1.0)
        errors[m] = round(err, 4)
        if err > tol:
            mismatch[m] = (y_m, y_e)
    return {"borne_out": not bool(mismatch), "errors": errors,
            "mismatch": mismatch, "tol": tol}


# ── 模型基规划 (state-space rollout, 对标模型驱动规划) ─────────────


class ModelBasedPlanner:
    """在状态空间里用世界模型 rollout, 产出"计划即数学".

    - 对每个初始状态 × 候选动作, predict 预告后继状态;
    - 按预测目标(取某指标的期望)排候选动作, 返回 Ranked PlanStep;
    - 预告只做决策依据, 落地仍由真实执行检验 (Planner 不代替执行).
    """

    def __init__(self, model: WorldModel, objective: str = "T_eq_K",
                 sense: str = "maximize") -> None:
        self.model = model
        self.objective = objective
        self.sense = sense  # "maximize" | "minimize"

    def rollout(self, state: WorldState,
                actions: list[Action]) -> list[PlanStep]:
        steps: list[PlanStep] = []
        for a in actions:
            pred = self.model.predict(state, a)
            steps.append(PlanStep(action=a, law=self.model.law(), predicted=pred))
        return steps

    def plan(self, state: WorldState, actions: list[Action]) -> list[PlanStep]:
        """按预测目标排序, 返回从优到劣的计划(每步都带数学定律与预告)."""
        steps = self.rollout(state, actions)
        sign = 1.0 if self.sense == "maximize" else -1.0
        steps.sort(key=lambda s: sign * s.predicted.get(self.objective, -math.inf),
                   reverse=True)
        return steps

    def best(self, state: WorldState, actions: list[Action]) -> PlanStep | None:
        steps = self.plan(state, actions)
        return steps[0] if steps else None