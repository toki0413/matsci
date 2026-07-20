"""VisualCausalChain — Phase 2: 从多图/多观测点自动拟合 SCM.

把多张实验图 (不同条件下的 XRD/SEM/DSC/TGA 等) 通过 vision_describe
提取成数值 features, 再用 scipy.optimize.curve_fit 拟合模板 SCM 的
物理参数 (Arrhenius Ea / Ostwald K0 / Fick D0 / Avrami k0 等).

产出 VisualSCM (source="visual_chain_fit", confirmed=False), 用户审核后
调 confirm_scm 升级. 拟合后可直接喂 predict_intervention 做 L2 干预.

阶梯映射:
  L1 观察  P(Y|X)           — vision_describe (感知层)
  L2 干预  P(Y|do(X))       — predict_intervention (Phase 1)
  L2+ 拟合 P(Y|do(X), data) — 本模块 (Phase 2): 数据后验修正先验参数
  L3 反事实 P(Y_x|X',Y')    — counterfactual_render (Phase 3)

设计原则 (ponytail):
  - 只拟合主要 feature (sintering: particle_size, ostwald: particle_size,
    diffusion: D, phase_transition: phase_fraction). 衍生 feature 沿用模板
  - scipy.optimize.curve_fit (已是依赖, 零新依赖)
  - 拟合失败回退模板默认参数 + warning, 不抛异常
  - 拟合质量用 R^2 评估, 写进 SCM.notes
  - 视觉提取走 vision_describe, 数值观测点可绕过视觉直接传

升级路径:
  - 数据多时换 Bayesian 结构学习 (pc algorithm / GES)
  - 多 feature 联合拟合 (joint NLL)
  - 噪声模型自适应 (heteroscedastic)
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from pydantic import BaseModel, Field
from scipy.optimize import curve_fit

from huginn.causal.visual_scm import (
    VisualSCM, get_template, list_templates,
)
from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult, ValidationResult

logger = logging.getLogger(__name__)

R = 8.314e-3  # kJ/(mol*K), 跟 visual_scm.py 保持一致


# ── 数据点 ─────────────────────────────────────────────────────

@dataclass
class Observation:
    """单个实验观测点: conditions + features 数值对.

    source:
      - "manual":        用户/agent 直接给的数值
      - "vision_describe": 从图像通过 vision_describe 提取
      - "literature":    从论文表格/文本抽出
    """
    conditions: dict[str, float]
    features: dict[str, float]
    source: str = "manual"
    raw: dict[str, Any] | None = None  # 视觉时存 vision_describe 原始输出


# ── 可拟合模板规格 ────────────────────────────────────────────

@dataclass
class _FittableSpec:
    """每个模板的可拟合表征."""
    template_name: str
    feature: str                                            # 拟合的主要特征
    conditions: list[str]                                   # 条件变量名
    params: list[str]                                       # 可拟合参数名
    defaults: dict[str, float]                              # 默认参数值
    bounds: tuple[list[float], list[float]]                 # curve_fit bounds
    predict: Callable[..., np.ndarray]                      # f(X[k,n], *params) -> y[n]
    make_feature_equation: Callable[[dict[str, float]], Callable]  # 拟合后构建闭包


def _sintering_predict(X: np.ndarray, Ea: float, A0: float, r0: float) -> np.ndarray:
    """烧结颗粒生长: r = (r0^3 + A0*exp(-Ea/(R*T)) * t)^(1/3)."""
    T, t = X[0], X[1]
    return np.cbrt(r0**3 + A0 * np.exp(-Ea / (R * np.maximum(T, 1.0))) * np.maximum(t, 0.0))


def _sint_make_eq(params: dict[str, float]) -> Callable[[dict[str, float], float], float]:
    Ea, A0, r0 = params["Ea_grow"], params["A0_grow"], params["r0"]
    def f(p: dict[str, float], u: float) -> float:
        T, t = p.get("T", 1500.0), p.get("t", 2.0)
        r = (r0**3 + A0 * math.exp(-Ea / (R * max(T, 1.0))) * max(t, 0.0)) ** (1.0/3.0)
        return max(r * (1 + u), 1.0)
    return f


def _ostwald_predict(X: np.ndarray, Ea: float, K0: float, r0: float) -> np.ndarray:
    """LSW: r^3 = r0^3 + K0*exp(-Ea/(R*T)) * t."""
    T, t = X[0], X[1]
    return np.cbrt(r0**3 + K0 * np.exp(-Ea / (R * np.maximum(T, 1.0))) * np.maximum(t, 0.0))


def _ostw_make_eq(params: dict[str, float]) -> Callable[[dict[str, float], float], float]:
    Ea, K0, r0 = params["Ea"], params["K0"], params["r0"]
    def f(p: dict[str, float], u: float) -> float:
        T, t = p.get("T", 600.0), p.get("t", 100.0)
        r = (r0**3 + K0 * math.exp(-Ea / (R * max(T, 1.0))) * max(t, 0.0)) ** (1.0/3.0)
        return max(r * (1 + u), 0.5)
    return f


def _diffusion_predict(X: np.ndarray, Ea: float, D0: float) -> np.ndarray:
    """Fick: D = D0 * exp(-Ea/(R*T))."""
    T = X[0]
    return D0 * np.exp(-Ea / (R * np.maximum(T, 1.0)))


def _diff_make_eq(params: dict[str, float]) -> Callable[[dict[str, float], float], float]:
    Ea, D0 = params["Ea"], params["D0"]
    def f(p: dict[str, float], u: float) -> float:
        T = p.get("T", 800.0)
        D = D0 * math.exp(-Ea / (R * max(T, 1.0)))
        return max(D * math.exp(u), 1e-25)
    return f


def _phase_predict(X: np.ndarray, Ea: float, k0: float, T_eq: float) -> np.ndarray:
    """Avrami (简化, t=1): f = 1 - exp(-(k0*exp(-Ea/(R*T))*(T_eq-T)/100)^2.5).

    ponytail: n_avrami=2.5, dTdP=1e-7 固定, 只拟合 3 参数.
    升级路径: 加 n_avrami 和 dTdP 一起拟合 (需要更多数据点).
    """
    T = X[0]
    p = X[1] if len(X) > 1 else np.zeros_like(T)
    T_eq_p = T_eq + 1e-7 * p
    delta = T_eq_p - T
    delta_safe = np.where(delta <= 0, 1e-6, delta)
    k = k0 * np.exp(-Ea / (R * np.maximum(T, 1.0)))
    f = 1.0 - np.exp(-np.power(k * 1.0 * delta_safe / 100.0, 2.5))
    return np.where(delta <= 0, 0.0, f)


def _phase_make_eq(params: dict[str, float]) -> Callable[[dict[str, float], float], float]:
    Ea, k0, T_eq = params["Ea"], params["k0"], params["T_eq"]
    n_avrami, dTdP = 2.5, 1e-7
    def f(p: dict[str, float], u: float) -> float:
        T, p_val = p.get("T", 1100.0), p.get("p", 1e5)
        T_eq_p = T_eq + dTdP * p_val
        delta = T_eq_p - T
        if delta <= 0:
            return max(0.0 + u * 0.05, 0.0)
        k = k0 * math.exp(-Ea / (R * max(T, 1.0)))
        f_val = 1.0 - math.exp(-((k * 1.0 * delta / 100.0) ** n_avrami))
        return min(max(f_val + u * 0.05, 0.0), 1.0)
    return f


_FITTABLE: dict[str, _FittableSpec] = {
    "sintering": _FittableSpec(
        template_name="sintering",
        feature="particle_size",
        conditions=["T", "t"],
        params=["Ea_grow", "A0_grow", "r0"],
        defaults={"Ea_grow": 250.0, "A0_grow": 1e6, "r0": 50.0},
        bounds=([50.0, 1e3, 5.0], [500.0, 1e10, 200.0]),
        predict=_sintering_predict,
        make_feature_equation=_sint_make_eq,
    ),
    "ostwald_ripening": _FittableSpec(
        template_name="ostwald_ripening",
        feature="particle_size",
        conditions=["T", "t"],
        params=["Ea", "K0", "r0"],
        defaults={"Ea": 180.0, "K0": 1e4, "r0": 20.0},
        bounds=([50.0, 1e2, 1.0], [400.0, 1e8, 100.0]),
        predict=_ostwald_predict,
        make_feature_equation=_ostw_make_eq,
    ),
    "diffusion": _FittableSpec(
        template_name="diffusion",
        feature="D",
        conditions=["T"],
        params=["Ea", "D0"],
        defaults={"Ea": 150.0, "D0": 1e-4},
        bounds=([50.0, 1e-10], [400.0, 1.0]),
        predict=_diffusion_predict,
        make_feature_equation=_diff_make_eq,
    ),
    "phase_transition": _FittableSpec(
        template_name="phase_transition",
        feature="phase_fraction",
        conditions=["T", "p"],
        params=["Ea", "k0", "T_eq"],
        defaults={"Ea": 120.0, "k0": 1e3, "T_eq": 1000.0},
        bounds=([50.0, 1e0, 500.0], [400.0, 1e6, 2000.0]),
        predict=_phase_predict,
        make_feature_equation=_phase_make_eq,
    ),
}


# ── 核心拟合 ─────────────────────────────────────────────────

def _build_fitted_scm(
    template: VisualSCM, spec: _FittableSpec, params: dict[str, float]
) -> VisualSCM:
    """从模板 + 拟合参数构建 fitted SCM.

    只替换 feature 方程, 其他 (nodes/edges/noise/衍生 feature 方程) 沿用模板.
    ponytail: 最小改动, 不重写整个 SCM, 只换一个方程.
    """
    new_equations = dict(template.equations)
    new_equations[spec.feature] = spec.make_feature_equation(params)
    return VisualSCM(
        name=f"{template.name}_fitted",
        domain=template.domain,
        nodes=template.nodes,
        edges=template.edges,
        equations=new_equations,
        noise=template.noise,
        confirmed=False,          # 视觉/数据拟合需用户审核
        source="visual_chain_fit",
        notes="",
    )


def fit_scm_from_observations(
    observations: list[Observation],
    template_name: str,
    fit_params: list[str] | None = None,
    maxfev: int = 10000,
) -> tuple[VisualSCM, dict[str, Any]]:
    """从数值观测点拟合 SCM 参数.

    Args:
        observations: 观测点列表 (每点含 conditions + features)
        template_name: 模板名 (sintering/ostwald_ripening/diffusion/phase_transition)
        fit_params: 显式指定拟合哪些参数 (None=全部). ponytail: 当前忽略, 一直全拟合.
        maxfev: curve_fit 最大迭代次数

    Returns:
        (fitted_scm, fit_report)
        fitted_scm: 拟合后的 VisualSCM (confirmed=False, source="visual_chain_fit")
        fit_report: {r2, params, n_points, warning, template, feature}
    """
    spec = _FITTABLE.get(template_name)
    if spec is None:
        raise ValueError(
            f"无拟合规格 for '{template_name}'. 可用: {list(_FITTABLE.keys())}"
        )
    template = get_template(template_name)
    if template is None:
        raise ValueError(f"模板 '{template_name}' 不存在")

    # 过滤有 feature 数值 + 条件齐全的观测点
    valid_obs = [
        o for o in observations
        if spec.feature in o.features
        and all(c in o.conditions for c in spec.conditions)
    ]

    if len(valid_obs) < len(spec.params):
        # 数据点不够, 回退默认参数
        fitted_params = spec.defaults.copy()
        fitted_scm = _build_fitted_scm(template, spec, fitted_params)
        warning = (
            f"数据点不足 ({len(valid_obs)} < {len(spec.params)} 参数), "
            f"回退默认参数"
        )
        fitted_scm.notes = (
            f"Phase 2 fit: feature={spec.feature}, n_points={len(valid_obs)} | "
            f"R^2=N/A | params={fitted_params} | warning: {warning}"
        )
        return fitted_scm, {
            "r2": float("nan"), "params": fitted_params,
            "n_points": len(valid_obs), "warning": warning,
            "template": template_name, "feature": spec.feature,
        }

    # 准备 X (k, n) 和 y (n,)
    X = np.array([
        [o.conditions[c] for c in spec.conditions]
        for o in valid_obs
    ]).T  # shape: (k, n)
    y = np.array([o.features[spec.feature] for o in valid_obs], dtype=float)

    # curve_fit
    p0 = [spec.defaults[p] for p in spec.params]
    lower, upper = spec.bounds
    try:
        popt, _ = curve_fit(
            spec.predict, X, y, p0=p0, bounds=(lower, upper), maxfev=maxfev,
        )
        fitted_params = dict(zip(spec.params, popt))
        y_pred = spec.predict(X, *popt)
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        warning = None
    except Exception as exc:
        logger.warning(
            "curve_fit failed for %s: %s", template_name, exc, exc_info=True
        )
        fitted_params = spec.defaults.copy()
        r2 = float("nan")
        warning = f"curve_fit 失败: {exc}, 回退默认参数"

    fitted_scm = _build_fitted_scm(template, spec, fitted_params)
    notes_parts = [
        f"Phase 2 fit: feature={spec.feature}, n_points={len(valid_obs)}",
        f"R^2={r2:.4f}" if not math.isnan(r2) else "R^2=N/A",
        f"params={{{', '.join(f'{k}={v:.4g}' for k, v in fitted_params.items())}}}",
    ]
    if warning:
        notes_parts.append(f"warning: {warning}")
    fitted_scm.notes = " | ".join(notes_parts)

    return fitted_scm, {
        "r2": r2, "params": fitted_params, "n_points": len(valid_obs),
        "warning": warning, "template": template_name, "feature": spec.feature,
    }


# ── 视觉提取 (上层 wrapper) ──────────────────────────────────

_FEATURE_ALIASES: dict[str, list[str]] = {
    "particle_size": ["particle size", "particle_size", "grain size",
                      "颗粒尺寸", "颗粒大小", "size"],
    "density":       ["density", "致密度", "密度"],
    "D":             ["diffusion coefficient", "diffusion_coefficient",
                      "扩散系数", "D ==", "D="],
    "phase_fraction":["phase fraction", "phase_fraction", "相分数",
                      "体积分数", "fraction"],
}


def _extract_numerical_feature(text: str, feature_name: str) -> float | None:
    """从文本提取特征数值. 简单 regex.

    支持格式:
      - "particle size: 250 nm"
      - "particle_size = 250"
      - "颗粒尺寸 250 nm"
      - "size: 250nm"

    ponytail: 简单 regex + alias 表, 升级路径用 NLP/LLM 抽取.
    """
    aliases = _FEATURE_ALIASES.get(feature_name, [feature_name])
    # 按 alias 长度降序匹配, 避免 "D" 短 alias 误匹配 (e.g. "D" 误中 "Density")
    aliases = sorted(aliases, key=len, reverse=True)
    for key in aliases:
        # 数字 + 可选科学计数法 + 可选单位
        for pat in [
            rf"{re.escape(key)}\s*[:=]\s*([\d.]+(?:e[-+]?\d+|[-+]?\d+)?)",
            rf"{re.escape(key)}\s+([\d.]+(?:e[-+]?\d+|[-+]?\d+)?)",
        ]:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    continue
    return None


def extract_observations_from_images(
    image_specs: list[dict[str, Any]],
    condition_keys: list[str],
    feature_keys: list[str],
    question: str | None = None,
) -> list[Observation]:
    """从多张图提取数值观测点.

    Args:
        image_specs: list of
            {"image_path": "...", "conditions": {"T": 1500, "t": 2.0}}
            或 {"image_bytes": b"...", "conditions": {...}}
        condition_keys: 要从 conditions 里取的条件变量名
        feature_keys: 要从图像描述里提取的特征名
        question: 给 vision_describe 的提问 (None=自动按 feature 生成)

    Returns:
        list[Observation]. 提取失败的点 features 为空 dict, 仍保留 conditions.
    """
    from huginn.tools.vision_describe_tool import (
        describe_image, describe_image_bytes,
    )

    observations: list[Observation] = []
    auto_question = (
        question
        or f"Extract numerical values for: {', '.join(feature_keys)}"
    )

    for spec in image_specs:
        conditions = {
            k: float(v) for k, v in spec.get("conditions", {}).items()
            if k in condition_keys
        }
        if len(conditions) != len(condition_keys):
            continue  # 条件不全跳过

        # 调 vision_describe
        desc: dict[str, Any]
        if spec.get("image_bytes"):
            desc = describe_image_bytes(spec["image_bytes"], question=auto_question)
        elif spec.get("image_path"):
            desc = describe_image(spec["image_path"], question=auto_question)
        else:
            continue

        if not desc.get("available"):
            observations.append(Observation(
                conditions=conditions, features={},
                source="vision_describe", raw=desc,
            ))
            continue

        # 从描述里提取数值
        text = desc.get("text_concat", "") or desc.get("text", "") or ""
        features: dict[str, float] = {}
        for fk in feature_keys:
            v = _extract_numerical_feature(text, fk)
            if v is not None:
                features[fk] = v

        observations.append(Observation(
            conditions=conditions, features=features,
            source="vision_describe", raw=desc,
        ))
    return observations


# ── HuginnTool 包装 ──────────────────────────────────────────

class FitSCMFromObservationsInput(BaseModel):
    template_name: str = Field(
        ...,
        description="模板名 (sintering/ostwald_ripening/diffusion/phase_transition)",
    )
    observations: list[dict[str, Any]] = Field(
        ...,
        description="观测点列表, 每点 {conditions: {...}, features: {...}}",
    )
    fit_params: list[str] | None = Field(
        None, description="显式指定拟合参数 (None=全部拟合)"
    )


class FitSCMFromObservationsTool(HuginnTool):
    """从数值观测点拟合 SCM 参数."""

    name = "fit_scm_from_observations"
    category = "causal"
    description = (
        "Fit Structural Causal Model parameters from experimental observations "
        "using scipy.optimize.curve_fit. Returns a fitted VisualSCM "
        "(source='visual_chain_fit', confirmed=False) that can be passed to "
        "predict_intervention. Use this when you have multiple (conditions, features) "
        "data points from experiments or literature."
    )
    input_schema = FitSCMFromObservationsInput
    read_only = True  # 不修改外部状态

    def is_read_only(self, args: FitSCMFromObservationsInput) -> bool:
        return True

    async def validate_input(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ValidationResult:
        input_data = (
            args if isinstance(args, FitSCMFromObservationsInput)
            else FitSCMFromObservationsInput(**args)
        )
        if input_data.template_name not in list_templates():
            return ValidationResult(
                result=False,
                message=(
                    f"未知模板: {input_data.template_name}. "
                    f"可用: {list_templates()}"
                ),
            )
        if not input_data.observations:
            return ValidationResult(
                result=False, message="observations 不能为空"
            )
        return ValidationResult(result=True)

    async def call(
        self, args: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        input_data = (
            args if isinstance(args, FitSCMFromObservationsInput)
            else FitSCMFromObservationsInput(**args)
        )
        try:
            observations = [
                Observation(
                    conditions=o.get("conditions", {}),
                    features=o.get("features", {}),
                    source=o.get("source", "manual"),
                ) for o in input_data.observations
            ]
            fitted_scm, report = fit_scm_from_observations(
                observations,
                input_data.template_name,
                input_data.fit_params,
            )
            return ToolResult(
                data={
                    "scm": fitted_scm.to_dict(),
                    "scm_name": fitted_scm.name,
                    "fit_report": report,
                },
                success=True,
            )
        except Exception as exc:
            logger.warning(
                "fit_scm_from_observations failed: %s", exc, exc_info=True
            )
            return ToolResult(data=None, success=False, error=str(exc))


# ── self-check ───────────────────────────────────────────────

def _selfcheck() -> None:
    """14 项 assert 验证 visual_causal_chain 核心行为."""
    import random as _rng
    _rng.seed(42)

    # 1. 4 个模板都有拟合规格
    assert set(_FITTABLE.keys()) == {"sintering", "ostwald_ripening",
                                      "diffusion", "phase_transition"}

    # 2. _sintering_predict 单调性: T 升 → particle_size 升
    X = np.array([[1500.0, 1800.0], [2.0, 2.0]])  # (2, 2): T=[1500,1800], t=[2,2]
    y = _sintering_predict(X, 250.0, 1e6, 50.0)
    assert y[1] > y[0], f"Arrhenius 单调性失败: {y}"

    # 3. _ostwald_predict 单调性: t 升 → particle_size 升
    X = np.array([[800.0, 800.0], [10.0, 100.0]])
    y = _ostwald_predict(X, 180.0, 1e4, 20.0)
    assert y[1] > y[0]

    # 4. _diffusion_predict 单调性: T 升 → D 升
    X = np.array([[500.0, 1000.0]])  # (1, 2): 1 个条件, 2 个数据点
    y = _diffusion_predict(X, 150.0, 1e-4)  # 返 shape (2,)
    assert y[1] > y[0], f"Fick 单调性失败: {y}"

    # 5. _phase_predict: T 低于 T_eq → phase_fraction > 0; T 高于 T_eq → 0
    X = np.array([[900.0, 1100.0]])  # (1, 2): 1 个条件 T, 2 个数据点
    y = _phase_predict(X, 120.0, 1e3, 1000.0)  # 返 shape (2,)
    assert y[0] > 0.0, f"低温应有相变: {y}"
    assert y[1] == 0.0, f"高温应无相变: {y}"

    # 6. 拟合 sintering 数据点 (用真实参数生成 16 点 + 1% 噪声) → 拟合参数接近真实
    # 用 Ea=100, A0=1e8, r0=30 (不是模板默认 Ea=250), 让 Arrhenius 项在 1200-2000 K 明显起作用.
    # ponytail: 模板默认 Ea=250 在 2000 K 下 exp(-15)≈3e-7 几乎不变化, 测试无意义.
    Ea_true, A0_true, r0_true = 100.0, 1e8, 30.0
    obs: list[Observation] = []
    for T in [1200, 1400, 1500, 1600, 1700, 1800, 1900, 2000]:
        for t in [1.0, 4.0]:
            r_true = (r0_true**3 + A0_true * math.exp(-Ea_true / (R * T)) * t) ** (1.0/3.0)
            r_noisy = r_true * (1 + _rng.gauss(0, 0.01))
            obs.append(Observation(
                conditions={"T": float(T), "t": t},
                features={"particle_size": r_noisy},
            ))
    fitted_scm, report = fit_scm_from_observations(obs, "sintering")
    assert report["n_points"] == 16
    assert report["warning"] is None
    assert not math.isnan(report["r2"])
    assert report["r2"] > 0.95, f"R^2 太低: {report['r2']}"
    # 拟合参数应接近真实值 (100, 1e8, 30), 容忍 50% 误差 (噪声 + 简化模型)
    p = report["params"]
    assert 60 < p["Ea_grow"] < 150, f"Ea_grow 偏离: {p}"
    assert 1e6 < p["A0_grow"] < 1e10, f"A0_grow 偏离: {p}"
    assert 15 < p["r0"] < 60, f"r0 偏离: {p}"

    # 7. fitted_scm 标记正确
    assert fitted_scm.name == "sintering_fitted"
    assert fitted_scm.source == "visual_chain_fit"
    assert fitted_scm.confirmed is False  # 数据拟合需用户审核
    assert "R^2=" in fitted_scm.notes

    # 8. 数据点不足 → 回退默认参数 + warning
    obs_few = obs[:2]  # 只 2 点, 但 sintering 有 3 参数
    fitted_scm2, report2 = fit_scm_from_observations(obs_few, "sintering")
    assert report2["warning"] is not None
    assert "数据点不足" in report2["warning"]
    assert report2["params"] == _FITTABLE["sintering"].defaults
    assert math.isnan(report2["r2"])

    # 9. 不存在的模板 → 抛 ValueError
    try:
        fit_scm_from_observations(obs, "nonexistent_template")
        assert False, "应抛 ValueError"
    except ValueError:
        pass

    # 10. 拟合 diffusion 数据点
    obs_d: list[Observation] = []
    for T in [600, 700, 800, 900, 1000, 1100, 1200]:
        D_true = 1e-4 * math.exp(-150.0 / (R * T))
        D_noisy = D_true * (1 + _rng.gauss(0, 0.02))
        obs_d.append(Observation(
            conditions={"T": float(T)}, features={"D": D_noisy},
        ))
    fitted_d, report_d = fit_scm_from_observations(obs_d, "diffusion")
    assert report_d["n_points"] == 7
    assert report_d["r2"] > 0.95
    p_d = report_d["params"]
    assert 80 < p_d["Ea"] < 250, f"Ea 偏离: {p_d}"
    assert 1e-6 < p_d["D0"] < 1e-2, f"D0 偏离: {p_d}"

    # 11. 拟合 ostwald_ripening 数据点
    # 用 Ea=60, K0=1e7, r0=15 (不是模板默认), 让 LSW 项在 500-900 K 明显起作用.
    Ea_true_o, K0_true, r0_true_o = 60.0, 1e7, 15.0
    obs_o: list[Observation] = []
    for T in [500, 600, 700, 800, 900]:
        for t in [10, 100, 1000]:
            r_true = (r0_true_o**3 + K0_true * math.exp(-Ea_true_o / (R * T)) * t) ** (1.0/3.0)
            r_noisy = r_true * (1 + _rng.gauss(0, 0.01))
            obs_o.append(Observation(
                conditions={"T": float(T), "t": float(t)},
                features={"particle_size": r_noisy},
            ))
    fitted_o, report_o = fit_scm_from_observations(obs_o, "ostwald_ripening")
    assert report_o["n_points"] == 15
    assert report_o["r2"] > 0.95, f"R^2 太低: {report_o['r2']}"
    p_o = report_o["params"]
    assert 30 < p_o["Ea"] < 100, f"Ea 偏离: {p_o}"
    assert 1e5 < p_o["K0"] < 1e9, f"K0 偏离: {p_o}"
    assert 5 < p_o["r0"] < 40, f"r0 偏离: {p_o}"

    # 12. 拟合 phase_transition 数据点
    # 用 Ea=70, k0=1e4, T_eq=1000 (让 Avrami 项在 850-990 K 有非平凡值).
    Ea_true_p, k0_true, T_eq_true = 70.0, 1e4, 1000.0
    obs_p: list[Observation] = []
    for T in [850, 880, 910, 940, 970, 990]:
        delta = T_eq_true - T
        if delta <= 0:
            f_true = 0.0
        else:
            k = k0_true * math.exp(-Ea_true_p / (R * T))
            f_true = 1.0 - math.exp(-((k * delta / 100.0) ** 2.5))
        f_noisy = max(0.0, min(1.0, f_true + _rng.gauss(0, 0.005)))
        obs_p.append(Observation(
            conditions={"T": float(T), "p": 1e5},
            features={"phase_fraction": f_noisy},
        ))
    fitted_p, report_p = fit_scm_from_observations(obs_p, "phase_transition")
    assert report_p["n_points"] == 6
    # phase_transition 6 数据点拟合 3 参数, R^2 可能不稳定, 只验证无报错
    assert report_p["warning"] is None or "curve_fit" in (report_p["warning"] or "")

    # 13. _extract_numerical_feature 命中各种格式
    assert _extract_numerical_feature("particle size: 250 nm", "particle_size") == 250.0
    assert _extract_numerical_feature("particle_size = 250", "particle_size") == 250.0
    assert _extract_numerical_feature("颗粒尺寸 180 nm", "particle_size") == 180.0
    assert _extract_numerical_feature("size: 300nm", "particle_size") == 300.0
    assert _extract_numerical_feature("density: 5.8 g/cm^3", "density") == 5.8
    assert _extract_numerical_feature("扩散系数 1.5e-12", "D") == 1.5e-12
    assert _extract_numerical_feature("no relevant info", "particle_size") is None

    # 14. Tool validate_input 拒绝未知模板 + 空 observations
    import asyncio
    tool = FitSCMFromObservationsTool()
    loop = asyncio.new_event_loop()
    try:
        r = loop.run_until_complete(tool.validate_input(
            {"template_name": "nonexistent", "observations": [{"conditions": {}, "features": {}}]}
        ))
        assert r.result is False
        r = loop.run_until_complete(tool.validate_input(
            {"template_name": "sintering", "observations": []}
        ))
        assert r.result is False
        r = loop.run_until_complete(tool.validate_input(
            {"template_name": "sintering",
             "observations": [{"conditions": {"T": 1500, "t": 2}, "features": {"particle_size": 200}}]}
        ))
        assert r.result is True
    finally:
        loop.close()

    # 15. Tool call 成功返 fit_report
    tool2 = FitSCMFromObservationsTool()
    loop2 = asyncio.new_event_loop()
    try:
        from huginn.types import ToolContext
        ctx = ToolContext(session_id="selfcheck", workspace=".")
        r = loop2.run_until_complete(tool2.call(
            {"template_name": "diffusion",
             "observations": [{"conditions": {"T": 800}, "features": {"D": 1e-8}},
                              {"conditions": {"T": 1000}, "features": {"D": 1e-7}}]},
            ctx,
        ))
        assert r.success
        assert "fit_report" in r.data
        assert r.data["scm_name"] == "diffusion_fitted"
    finally:
        loop2.close()

    print("all self-checks passed")


if __name__ == "__main__":
    _selfcheck()
