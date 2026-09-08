"""科学定律模型核心 (LawModel) —— 对标 世界模型/VLA/机器人分层规划, 以**数学定律**为
规划与验证的载体. 科研级前向模型 + 可证伪验证.

设计说明 (与既有实现的**关系**, 避免命名混淆): 
  仓库已有 security/world_state.py(ObsVector/StateEstimator/**ForwardPredictor** 前向
  投影, 服务 agent 控制环/奖励/记忆) 与 security/world_model.py(物理动作**逆生成器**,
  服务可逆撤销). 本模块不再叫 world_model/world_state 以免撞名, 改称 **LawModel**, 只
  做科研管线的**第一性原理前向模型 + 数学对账**:
    - 与 ForwardPredictor 的差别: 我提供 `law()`(数学定律方程串) 作为规划/验证的公共
      语言, 并有 `reconcile()` **可证伪对账**(borne_out/falsified) —— 前向投影只报告
      "下一状态", 不校验"定律是否被执行证实"; 这是科研结论的硬验证, 二者互补而非替换.
    - 消费方: 科研管线 / ScienceTeam(而非 sandbox 控制环).

核心对象:
  - :class:`LawState`  世界可测状态(科学系统的数值特征向量, 如 a_AU/S/T_eq)。
  - :class:`LawAction` 可执行动作(实验/仿真配置, 如轨道半长轴缩放 a_scale)。
  - :class:`LawModel`  世界转移模型 —— ``predict(state, action) -> state'``, 其 ``law()``
    返回该域的**数学定律(方程串)**。这是"模型基规划"里的 forward model。
  - :func:`reconcile`  预测 vs 真实执行的数值对账: 判定定律被证实/证伪 (偏差不覆盖)。
  - :class:`ModelBasedPlanner` 在状态空间里 rollout: 按预测目标排候选动作, 产出
    **计划即数学**(每步 PlanStep = 动作 + 定律 + 预告状态)。

诚实边界: 世界模型是**假说**, execute 是**真相检验**。predict 只预告, 不替代执行;
reconcile 把预测与真值并排审计, 不符则如实标 falsified。
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# 基础物理常数 (世界模型的数学参数) —— SI 单位
SOLAR_LUM = 3.828e26          # L☉ [W]
STEFAN = 5.670374419e-8        # σ [W/m²/K⁴]
AU_M = 1.49597870700e11        # [m]
SOLAR_MASS_KG = 1.98892e30


class Worldview(str, Enum):
    """世界模型的世界观谱系 (多元论治理) —— 不同学科对『世界/状态/动力学』的不同刻画.

    '世界模型'不是边界明确的单一技术范式: 强化学习、生成模型、认知科学、具身智能、
    复杂系统对'世界''状态'及其动力学各有理解。能力治理要求每个世界模型能力**显式声明
    自己站在哪一极**, 避免把"预测感知信号""学隐空间状态转移""刻画行动/物理约束/因果"
    混为一谈:

    - PHYSICS_CAUSAL:        把世界刻画为**行动/物理约束/因果**(方程驱动前向模型 +
                             真实执行对账)。这是 LawModel 的实现极: 预告(方程)→真实
                             执行→reconcile 数值对账(证伪式可信)。
    - PERCEPTION_PREDICTIVE: 世界=感官信号序列, 模型预测下一个感知信号 (感知闭环)。
    - LATENT_TRANSITION:     世界=隐空间状态, 学 s_{t+1}=f(s_t,a_t) 的转移。
    - BEHAVIOR_POLICY:       不显式建模世界, 直接从(状态,动作)学策略/价值 (RL 极)。
    """
    PHYSICS_CAUSAL = "physics_causal"
    PERCEPTION_PREDICTIVE = "perception_predictive"
    LATENT_TRANSITION = "latent_transition"
    BEHAVIOR_POLICY = "behavior_policy"


def world_model_card(model: "LawModel") -> dict:
    """世界模型能力的治理卡片 (多元论 + 具身可信).

    回答两个治理问题:
      - 它在世界模型谱系里**站在哪一极**(worldview)—— 防止"预测感知/隐态转移/物理因果"
        被当成同一类能力;
      - 它的预告是否**可证伪**(falsifiable)—— 真相检验由"真实执行"经 reconcile 提供,
        不可证伪的预判不得冒充"世界知识"。

    Returns: {"model", "domain", "worldview", "falsifiable", "truth_reference", "methods"}
    """
    return {
        "model": type(model).__name__,
        "domain": getattr(model, "domain", ""),
        "worldview": getattr(model, "worldview", Worldview.PHYSICS_CAUSAL).value,
        "falsifiable": (hasattr(model, "predict") and hasattr(model, "law")
                        and hasattr(model, "seed")),
        "truth_reference": "real_execution (reconcile 数值对账)",
        "methods": sorted(k for k in ("predict", "law", "seed") if hasattr(model, k)),
    }


def kepler_semimajor_au(period_d: float, host_msun: float = 1.0) -> float:
    """开普勒第三定律: a[AU] = (M*[M☉])^{1/3} (P[yr])^{2/3}. (数学定律)"""
    return host_msun ** (1 / 3) * (period_d / 365.25) ** (2 / 3)


# ── 状态 / 动作 (形式化的世界) ──────────────────────────────────


@dataclass
class LawState:
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
class LawAction:
    """可执行动作: 实验/仿真配置 (参数向量)."""
    config: dict[str, float] = field(default_factory=dict)
    label: str = ""

    def as_dict(self) -> dict:
        return {"label": self.label, "action": dict(self.config)}


@dataclass
class PlanStep:
    """计划即数学: 一个动作 + 其数学定律 + 定律预告的下继状态."""
    action: LawAction
    law: str = ""                        # 数学语言核心: 该步依据的方程
    predicted: LawState | None = None    # predict(state, action)

    def to_dict(self) -> dict:
        return {
            "action": self.action.as_dict(),
            "law": self.law,
            "predicted": self.predicted.as_dict() if self.predicted else None,
        }


# ── 定律模型 (forward model) ────────────────────────────────────


class LawModel(ABC):
    """世界的转移模型: predict(state, action) -> next state, law() 给数学方程. """

    domain = ""
    # 多元论治理: 本实现站在"物理-行动-因果"这一极 (见 Worldview). 其它极在能力面
    # 显式标注, 不混淆.
    worldview: Worldview = Worldview.PHYSICS_CAUSAL

    @abstractmethod
    def law(self) -> str:
        """返回该世界的数学定律(方程串)—— 规划/预告/验证共享的证据来源."""

    @abstractmethod
    def predict(self, state: LawState, action: LawAction) -> LawState:
        """预告: 施加动作后世界应转移到的状态 (数学定律的数值实现)."""

    @abstractmethod
    def seed(self, observation: dict[str, Any]) -> LawState:
        """从外部观测(读入的科学数据)落下初始状态."""


class FirstPrinciplesLawModel(LawModel):
    """系外行星第一性原理定律模型: 轨道日晒 - 黑体平衡温度.

    law (数学):

        a = (M*/M☉)^{1/3} (P/yr)^{2/3} [AU]           开普勒第三定律
        S = L☉ / (4π a²)  [W/m²]                      日晒(各向同性辐射)
        T_eq = ((1-α) S / (4σ))^{1/4}  [K]             黑体平衡温度(Stefan-Boltzmann)

    动作空间: LawAction.config = {"a_scale": s}   → 半长轴缩放 a' = a·s.
    """

    domain = "exoplanet"

    def __init__(self, albedo: float = 0.1, host_msun: float = 1.0) -> None:
        self.albedo = albedo
        self.host_msun = host_msun

    def law(self) -> str:
        return ("T_eq = ((1-α)·S/(4σ))^{1/4};  S = L☉/(4πa²);  "
                f"a = ({self.host_msun} M☉)^{1/3}(P/yr)^{{2/3}} AU;  α={self.albedo}")

    def seed(self, observation: dict[str, Any]) -> LawState:
        p_d = float(observation["orbper_d"])
        a_au = kepler_semimajor_au(p_d, self.host_msun)
        return LawState({"P_d": p_d, "a_AU": a_au}, domain=self.domain)

    def predict(self, state: LawState, action: LawAction) -> LawState:
        a_au = state.get("a_AU", 1.0) * action.config.get("a_scale", 1.0)
        a_m = a_au * AU_M
        S = SOLAR_LUM / (4 * math.pi * a_m ** 2)
        T_eq = ((1 - self.albedo) * S / (4 * STEFAN)) ** 0.25
        return LawState({"a_AU": round(a_au, 4),
                         "S_Wm2": round(S, 1),
                         "T_eq_K": round(T_eq, 1)}, domain=self.domain)


class MechanicsLawModel(LawModel):
    """简单谐振子 —— 第二类第一性原理域 (纯力学, 对标 VLA/机器人周期动力学).

    与 FirstPrinciplesLawModel(轨道热力学) 同构但属不同物理, 说明 LawModel 是
    **域无关**的: 世界怎样转移用数学定律描述, 规划/预告/验证共用同一套语言。

    law (数学):

        ω = √(k/m)            固有角频率 (刚度↑ / 质量↓ → ω↑)
        T = 2π/ω = 2π·√(m/k)  周期 (动能-势能相互转换的固有节律)

    动作空间: LawAction.config = {"k_scale": s_k, "m_scale": s_m}
      → 施加动作后 ω' = √((k·s_k)/(m·s_m)).

    域解读 (VLA/机器人对齐): 状态即"系统的力学特征"(质量、刚度), 动作即"改配置",
    定律预告"改变配置后系统的固有节律", 真实执行校验定律是否被证实。
    """

    domain = "mechanics"

    def law(self) -> str:
        return "ω = √(k/m);  T = 2π/ω = 2π·√(m/k)"

    def seed(self, observation: dict[str, Any]) -> LawState:
        return LawState({"mass_kg": float(observation["mass_kg"]),
                         "stiffness_Nm": float(observation["stiffness_Nm"])},
                        domain=self.domain)

    def predict(self, state: LawState, action: LawAction) -> LawState:
        m = state.get("mass_kg", 1.0) * action.config.get("m_scale", 1.0)
        k = state.get("stiffness_Nm", 1.0) * action.config.get("k_scale", 1.0)
        omega = math.sqrt(k / m)                     # ω = √(k/m)
        T = 2 * math.pi / omega                      # T = 2π/ω
        return LawState({"mass_kg": round(m, 4), "stiffness_Nm": round(k, 3),
                         "omega_rad_s": round(omega, 4), "T_s": round(T, 4)},
                        domain=self.domain)


# ── 预测 vs 真实 的数学对账 (可证伪性校验) ──────────────────────


def reconcile(predicted: LawState, actual: dict, *, tol: float = 0.03,
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
    """在状态空间里用定律模型 rollout, 产出"计划即数学".

    - 对每个初始状态 × 候选动作, predict 预告后继状态;
    - 按预测目标(取某指标的期望)排候选动作, 返回 Ranked PlanStep;
    - 预告只做决策依据, 落地仍由真实执行检验 (Planner 不代替执行).
    """

    def __init__(self, model: LawModel, objective: str = "T_eq_K",
                 sense: str = "maximize") -> None:
        self.model = model
        self.objective = objective
        self.sense = sense  # "maximize" | "minimize"

    def rollout(self, state: LawState,
                actions: list[LawAction]) -> list[PlanStep]:
        steps: list[PlanStep] = []
        for a in actions:
            pred = self.model.predict(state, a)
            steps.append(PlanStep(action=a, law=self.model.law(), predicted=pred))
        return steps

    def plan(self, state: LawState, actions: list[LawAction]) -> list[PlanStep]:
        """按预测目标排序, 返回从优到劣的计划(每步都带数学定律与预告)."""
        steps = self.rollout(state, actions)
        sign = 1.0 if self.sense == "maximize" else -1.0
        steps.sort(key=lambda s: sign * s.predicted.get(self.objective, -math.inf),
                   reverse=True)
        return steps

    def best(self, state: LawState, actions: list[LawAction]) -> PlanStep | None:
        steps = self.plan(state, actions)
        return steps[0] if steps else None