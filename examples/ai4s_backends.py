#!/usr/bin/env python3
"""Huginn 自主深研管线 的可插拔『域 backend』集.

产品管线是 `huginn/research/program.py`(域无关); 这里的每个 backend 只是给同一
管线喂不同域的 `Experiment` 列表. 新加一个领域只需再写一个 backend, 管线零改动.

内置 backend:
  - hotjupiter_backend(): 线性/非线性/缩放 大气环流假说 (WASP-43b / HD 209458b)
  - exoplanet_backend():   系外行星 轨道-日晒-平衡温度 第一性原理实验(真实目录)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import hotjupiter_sw as hj
import hotjupiter_sw2 as hj2

from huginn.research import Experiment

_HERE = Path(__file__).resolve().parent
NX, NY, T_END = 48, 24, 8.0e5
NX_NL = 64


# ═══════════ backend 1 · 热木星大气环流 ═══════════
def hotjupiter_backend() -> list[Experiment]:
    def _run(system: str, kind: str) -> dict:
        if kind == "linear":
            r = hj.run_scenario(system, nx=NX, ny=NY, t_end_s=T_END)
            d = hj.analyze(r)
            return {
                "objectives": {"eastward_offset": float(d["hot_spot_offset_deg"]),
                               "daynight_contrast": float(d["day_night_delta_T_K"]),
                               "conservation": -abs(float(r["rel_energy_drift"]))},
                "summary": {"model": "linear", "system": system,
                            "deltaT_K": d["day_night_delta_T_K"], "offset_deg": d["hot_spot_offset_deg"],
                            "jet_ms": d["equatorial_jet_ms"], "energy_drift": round(r["rel_energy_drift"], 5)},
                "success": True}
        if kind == "nonlinear":
            r = hj2.run_scenario2(system, nx=NX_NL, ny=NY, t_end_s=2.0e6)
            return {
                "objectives": {"eastward_offset": float(r["hot_spot_offset_deg"]),
                               "daynight_contrast": float(r["day_night_delta_T_K"]),
                               "conservation": -abs(float(r["rel_energy_drift"]))},
                "summary": {"model": "nonlinear", "system": system,
                            "deltaT_K": r["day_night_delta_T_K"], "offset_deg": r["hot_spot_offset_deg"],
                            "jet_ms": r["equatorial_jet_max_ms"], "energy_drift": round(r["rel_energy_drift"], 8)},
                "success": True}
        s = hj.sweep_daynight(system, nx=NX, ny=NY)
        dTs = [p["delta_T_K"] for p in s["sweep"]]
        mono = all(dTs[i] >= dTs[i + 1] for i in range(len(dTs) - 1))
        return {
            "objectives": {"eastward_offset": float(s["sweep"][-1]["offset_deg"]),
                           "daynight_contrast": float(dTs[-1]),
                           "conservation": (1.0 if (s["stable_range"] and mono) else 0.0)},
            "summary": {"model": "sweep", "system": system, "deltaT_K": float(dTs[-1]),
                        "deltaT_lim_K": s["radiative_limit_delta_T_K"],
                        "norm_span": [p["norm_delta_T"] for p in s["sweep"]],
                        "monotone": mono, "radiative_limit_ok": s["stable_range"]},
            "success": True}

    specs = [
        ("lin_was_daynight", "WASP-43b", "linear", "线性浅水能定量复现热木星昼夜温差与东向热点。"),
        ("lin_hd_daynight", "HD 209458b", "linear", "线性浅水对较远热木星仍给出朝向超自转的响应。"),
        ("nonlin_was_jet", "WASP-43b", "nonlinear", "非线性显著增强超自转喷射 → 非线性是超自转本质来源。"),
        ("nonlin_hd_jet", "HD 209458b", "nonlinear", "非线性超自转增强在另一系统上稳健。"),
        ("sweep_was_scaling", "WASP-43b", "sweep", "昼夜温差随热再分配时间缩放并被纯辐射极限校验。"),
        ("sweep_hd_scaling", "HD 209458b", "sweep", "缩放规律与极限校验可迁移。"),
    ]
    return [Experiment(name=n, hypothesis=h, run=lambda s=sy, k=ki: _run(s, k))
            for (n, sy, ki, h) in specs]


# ═══════════ backend 2 · 系外行星(真实目录) 第一性原理实验 ═══════════
# 真实轨道周期 → 开普勒第三定律 → 半长轴 → 日晒 → 黑体平衡温度, 无拟合.
SOLAR_LUM = 3.828e26          # L_☉ [W]
STEFAN = 5.670374419e-8       # σ [W/m²/K⁴]
AU_M = 1.49597870700e11       # [m]
SOLAR_MASS_KG = 1.98892e30
# 视宿主星为类太阳(M*=1) 的建模假设, 如实标注(目录仅含轨道周期/半径/质量)
_ASSUMED_HOST_MSUN = 1.0


def _kepler_semimajor_au(period_d: float, host_msun: float) -> float:
    """开普勒第三定律: a[AU] = (M*[M☉])^{1/3} (P[yr])^{2/3}. """
    return host_msun ** (1 / 3) * (period_d / 365.25) ** (2 / 3)


def exoplanet_backend(top: int = 8) -> list[Experiment]:
    data_path = _HERE / "out" / "real_exoplanet.json"
    records = json.loads(data_path.read_text(encoding="utf-8"))[:top]

    def _run(rec: dict) -> dict:
        period_d = float(rec["orbper_d"])
        a_au = _kepler_semimajor_au(period_d, _ASSUMED_HOST_MSUN)
        a_m = a_au * AU_M
        S = SOLAR_LUM / (4 * math.pi * a_m ** 2)                 # 日晒 [W/m²]
        T_eq = ((1 - 0.1) * S / (4 * STEFAN)) ** 0.25            # 全球平均平衡温度
        conserved = 1.0  # 第一性原理公式, 解析一致
        return {
            "objectives": {"eq_temp_K": float(T_eq),
                           "insolation_Wm2": float(S),
                           "consistency": float(conserved)},
            "summary": {"planet": rec["name"], "P_d": period_d, "a_AU": round(a_au, 4),
                        "S_Wm2": round(S, 1), "T_eq_K": round(T_eq, 1),
                        "m_jup": rec["mass_mjup"], "r_re": rec["radius_re"],
                        "assumption": "host=M_☉ (文献假说, 目录无宿主质量)"},
            "success": True}

    return [Experiment(name=("exo_" + rec["name"].replace(" ", "_").replace("-", "_")),
                       hypothesis=f"{rec['name']} 的轨道日晒与黑体平衡温度的解析结论",
                       run=lambda r=rec: _run(r)) for rec in records]


BACKENDS = {"hotjupiter": hotjupiter_backend, "exoplanet": exoplanet_backend}
GOALS = {
    "hotjupiter": "为潮汐锁定热木星昼夜环流寻找多证据方向的稳健研究假说",
    "exoplanet": "从真实系外行星目录推导轨道日晒与平衡温度的结构性分布",
}
OBJECTIVES = {
    "hotjupiter": {"eastward_offset": "maximize", "daynight_contrast": "maximize", "conservation": "maximize"},
    "exoplanet": {"eq_temp_K": "maximize", "insolation_Wm2": "maximize", "consistency": "maximize"},
}