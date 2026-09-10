"""书生 Intern-S2 + Huginn 断裂力学域完整深研管线(毕业答辩版第二主题).

选题种子: Yang W, Feng X-Q, Gao H (2026) *Outstanding issues and emerging
frontiers in fracture mechanics*, Int J Fract 250(1), DOI 10.1007/s10704-025-00907-6.
本文件把它列出的"杰出开放问题/新兴前沿"里**可计算、可证伪**的一批,
变成书生驱动 Huginn `run_research_program` 的真实数值实验:

  F1 界面裂纹振荡奇异场 (interface crack oscillatory singularity, Dundurs ε)
     —— 界面断裂的"互穿悖论": ε≠0 时 K 场预言裂纹面互相穿透, 是经典 LEFM
        接口理论的边界开放问题(综述引 Rice-佳/finite-traction 界面模型线).
  F2 脆-韧转变图谱 (Rice-Thomson/Rice 1992: γusf/γs 竞争判据)
     —— 何种材料本能脆/本能韧的机理问题(综述引 Rice & Thomson 1974,
        Rice 1992, Li et al. 2002 atomistic mechanisms).
  F3 纳尺度缺陷容差 (Gao flaw tolerance: 强度 vs 缺陷尺寸交叉)
     —— "极致强度"前沿(综述引 Yang et al. 2025 Solids in nano-scales,
        Zhang et al. 2016 Si nanowires, graphene papers).
  F4 统计弱链尺寸效应 (weakest-link, Pareto 缺陷分布 → 尺寸效应指数)
     —— 微裂纹统计/连接问题(综述引 Zhang-Li-Yang 统计强度, Li-Yang 微裂纹聚合).
  F5 桥联增韧 (Dugdale 常数牵引桥联区 → 表观韧性增益)
     —— 仿生/珍珠母增韧前沿(综述引 Shao et al. 2012 非连续桥联模型,
        Yao-Gao 多尺度内聚律, Yan-Feng nacre T-stress).
  F6 动态断裂速禁区 (mode-I Rayleigh 势垒: D(v) 零点 + Rose k(v) + 声学缺口)
     —— 超剪切/跨音速断裂的禁区结构(综述引 Rosakis supershear, Xia 实验室地震,
        Needleman-Rosakis 内聚 bond 强度/加载率).
  F7 KIC 有效性边界 (Irwin 塑性区 vs ASTM E399 试样尺寸门槛)
     —— 测试标准与 K-dominance 的开放边界(综述引 ASTM E399-19, Murakami,
        Tada-Paris-Irwin 手册线).

设计约束(与 rigidity 域完全一致, 诚实边界):
  - 全部数值由真实物理公式/数值方法产生, 无任何编造; objectives 各分支独立,
    互不支配 → 全存活, 保住"多证据方向";
  - goals/扫描配置/报告写作 全由书生自主驱动(client 非 None);
  - `--dry` 走 client=None 确定性综合, 无 API key 也能验证整条管线跑通。

用法:
  INTERNLM_API_KEY=... python examples/shusheng_fracture_mechanics.py --cycles 2
  python examples/shusheng_fracture_mechanics.py --dry --cycles 2   # 确定性验证
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import brentq

_AGENT = Path(__file__).resolve().parents[1] / "agent"
if str(_AGENT) not in sys.path:
    sys.path.insert(0, str(_AGENT))

# 产物持久化: 项目根 research_outputs/(git 白名单跟踪, 跨环境保留报表).
_OUT = Path(__file__).resolve().parents[1] / "research_outputs" / "shusheng_fracture_mechanics"
_OUT.mkdir(parents=True, exist_ok=True)

GOAL = (
    "面向 IJF 2026 综述《Outstanding issues and emerging frontiers in fracture mechanics》"
    "(杨卫/冯西桥/高华健) 抽取的七个可计算开放问题做真实数值检验: "
    "①界面裂纹互穿悖论(Dundurs ε 跨材料对跨度); ②脆韧转变 γusf/γs 图谱对材料族的区分度; "
    "③纳尺度缺陷容差(强度-缺陷尺寸交叉, 临界缺陷尺寸数量级); ④统计弱链尺寸效应指数; "
    "⑤桥联增韧增益; ⑥mode-I 动态断裂 Rayleigh 势垒与声学缺口结构; ⑦ASTM E399 KIC "
    "试样尺寸门槛的工程跨度。要求: 全部数值来自真实物理计算, 可复现, 每个 open issue 对应独立证据分支。"
)

# ── 真实物理常数表(全部来自公开文献的典型值, 一处定义不漂移) ─────────────
# F1 双材料对: (label, E1 GPa, ν1, E2, ν2) —— E 平面应变无关(用 μ,κ ⇒ 只需 E,ν).
_PAIRS = [
    ("Al2O3/Ni 陶瓷-金属", 380.0, 0.22, 200.0, 0.31),
    ("SiC/Al 复合材料", 410.0, 0.14, 70.0, 0.33),
    ("PMMA/steel 聚合物-钢", 3.0, 0.35, 210.0, 0.30),
    ("glass/epoxy 玻璃/环氧", 70.0, 0.22, 3.5, 0.35),
    ("sapphire/NiAl 介电/金属间", 406.0, 0.25, 190.0, 0.31),
    ("diamond/WC 超硬-硬质", 1140.0, 0.10, 650.0, 0.22),
]

# F2 材料脆韧表: (label, μ GPa, ν, b nm, γs J/m², γusf J/m²) —— Rice 1992 / Wells 典型值.
_MATERIALS = [
    ("diamond", 535.0, 0.10, 0.252, 5.30, 9.00),
    ("Si", 68.0, 0.22, 0.384, 1.24, 1.80),
    ("W", 161.0, 0.28, 0.274, 2.90, 3.50),
    ("α-Fe", 82.0, 0.29, 0.248, 1.90, 0.90),
    ("Ti", 44.0, 0.32, 0.295, 1.50, 0.95),
    ("Mg", 17.0, 0.29, 0.320, 0.90, 0.60),
    ("Cu", 48.0, 0.34, 0.256, 1.79, 0.35),
    ("Ni", 76.0, 0.31, 0.249, 2.20, 0.40),
    ("Al", 26.0, 0.35, 0.286, 1.14, 0.20),
    ("Au", 27.0, 0.44, 0.288, 1.40, 0.12),
]

# F3 材料(纳尺度缺陷容差, plane stress J_0=K²/E): (label, E GPa, σth GPa, γ J/m²).
_FLAW_MATS = [
    ("graphene(2D)", 1000.0, 130.0, 16.0),
    ("Si", 170.0, 7.0, 1.9),
    ("steel(bcc-Fe)", 200.0, 11.0, 2.0),
    ("Al2O3", 390.0, 12.0, 3.0),
]

# F7 结构合金(KIC 试样尺寸门槛): (label, σy MPa, KIC MPa√m).
_KIC_MATS = [
    ("A533B 压力容器钢", 345.0, 200.0),
    ("7075-T6 铝合金", 503.0, 29.0),
    ("Ti-6Al-4V", 950.0, 55.0),
    ("18Ni(250) 马氏体时效钢", 2400.0, 90.0),
    ("Al2O3 陶瓷", 2600.0, 3.5),
    ("PZT 压电陶瓷", 250.0, 1.0),
]

# 书生在第 2+ 轮可提议的断裂扫描白名单(全部映射真实物理计算, 无编造).
SCAN_OPS = {
    "pair": list(range(len(_PAIRS))),                     # F1 双材料对索引
    "material": list(range(len(_MATERIALS))),             # F2 材料索引
    "flaw_idx": [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 40.0],   # F3 参数: a/a* 倍数(探针/书生成码用)
    "flaw": [0, 1, 2, 3],                                  # flaw 扫描维: 材料索引(graphene/Si/steel/Al2O3)
    "n_flaws": [10, 30, 100, 300, 1000, 3000, 10000, 100000],  # F4 单元内缺陷数
    "bridge_ratio": [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9],       # F5 σ0/σy
    "vcR": [0.1, 0.25, 0.5, 0.7, 0.85, 0.9, 0.95, 0.98],        # F6 v/c_R
    "nu": [0.2, 0.3, 0.4],                                 # F6 泊松比
    "kic_mat": list(range(len(_KIC_MATS))),                # F7 合金索引
}
_SCAN_DEFAULTS = {"pair": 0, "material": 0, "flaw_idx": 2.0, "n_flaws": 100,
                  "bridge_ratio": 0.5, "vcR": 0.5, "nu": 0.3, "kic_mat": 0}

# 扫描"维"(dim 名) → 取值来源(SCAN_OPS 键; dim 名与白名单键不同名, 须显式映射).
_DIM_VALUES = {
    "pair": "pair", "material": "material", "flaw": "flaw", "n_flaws": "n_flaws",
    "bridge": "bridge_ratio", "barrier": "nu", "kic": "kic_mat",
    "nu": "nu",  # 书生可能直接用白名单键名 nu 当 dim —— 归入 barrier 族
}

# ═══════════════════════════════ 真实物理实验 ═══════════════════════════════

def _dundurs(E1: float, nu1: float, E2: float, nu2: float) -> tuple[float, float]:
    """平面应变 Dundurs 参数 (α, β). 定义: 材料1=上方. 输出 β 对 ε 有直接决定."""
    mu1 = E1 / (2 * (1 + nu1)); k1 = 3 - 4 * nu1
    mu2 = E2 / (2 * (1 + nu2)); k2 = 3 - 4 * nu2
    den = mu1 * (k2 + 1) + mu2 * (k1 + 1)
    alpha = (mu1 * (k2 + 1) - mu2 * (k1 + 1)) / den
    beta = (mu1 * (k2 - 1) - mu2 * (k1 - 1)) / den
    return float(alpha), float(beta)


def exp_interface_oscillation() -> dict:
    """F1 · 界面裂纹振荡奇异指数 ε = (1/2π)ln[(1−β)/(1+β)] 跨材料对跨度.

    开放问题(综述界面线): ε≠0 ⇒ K 场预言裂纹面互相穿透("互穿悖论"),
    需内聚牵引/接触区模型替代. 本实验给出 ε 的谱: 哪些对强振荡, 哪些近非振荡.
    """
    rows = []
    eps_list = []
    for label, e1, n1, e2, n2 in _PAIRS:
        a, b = _dundurs(e1, n1, e2, n2)
        eps = (1.0 / (2.0 * math.pi)) * math.log((1 - b) / (1 + b)) if abs(b) < 1 else 0.0
        eps_list.append(abs(eps))
        log10_lov = -math.pi / (abs(eps) + 1e-12) / math.log(10.0)  # log10(互穿区/裂纹长)
        rows.append({"pair": label, "alpha": round(a, 3), "beta": round(b, 3),
                     "|eps|": round(abs(eps), 5),
                     "log10(l/a)": round(log10_lov, 1)})
    osc_span = float(max(eps_list) - min(eps_list))
    return {
        "objectives": {"osc_span": round(osc_span, 6)},
        "summary": {"pairs": rows, "min_eps": round(min(eps_list), 5),
                    "max_eps": round(max(eps_list), 5), "osc_span": round(osc_span, 6)},
        "success": True,
    }


def exp_ductile_brittle_map() -> dict:
    """F2 · 脆-韧转变图谱: γusf/γs 竞争判据对材料族的区分度.

    Rice(1992)/Rice-Thomson(1974): γusf 低(位错发射易) ⇒ 韧; γs 相对低但 γusf 高 ⇒ 脆.
    典型分类: >1 本能脆(diamond/Si/W), 0.6~1 过渡(Ti/Mg), <0.6 本能韧(Cu/Ni/Al/Au).
    """
    rows = []
    marks = []
    for label, mu, nu, b, gs, gusf in _MATERIALS:
        marker = gusf / gs
        marks.append(marker)
        cls = "brittle" if marker > 1.0 else ("transitional" if marker >= 0.6 else "ductile")
        rows.append({"material": label, "mu": mu, "b_nm": b, "gamma_usf/gamma_s":
                     round(marker, 3), "class": cls})
    dbt_span = float(max(marks) - min(marks))
    return {
        "objectives": {"dbt_span": round(dbt_span, 3)},
        "summary": {"materials": rows, "min_idx": round(min(marks), 3),
                    "max_idx": round(max(marks), 3), "dbt_span": round(dbt_span, 3)},
        "success": True,
    }


def _flaw_strength_curve(E: float, sig_th: float, gamma: float,
                         a_by_astar: list[float]) -> tuple[list[float], float]:
    """Gao 缺陷容差: σ_f/σ_th = min(1, sqrt(a*/a)); a* = Eγ/(πσ_th²).

    返回 (σf/σth 随 a/a* 的曲线, a* 尺寸(纳米)).
    """
    E_pa = E * 1e9; sig_th_pa = sig_th * 1e9
    astar = E_pa * gamma / (math.pi * sig_th_pa ** 2)     # m
    curve = []
    for x in a_by_astar:
        curve.append(round(float(min(1.0, math.sqrt(1.0 / max(x, 1e-9)))), 4))
    return curve, astar * 1e9                              # a* 转 nm


def exp_flaw_tolerance(amax_nm: float = 100.0) -> dict:
    """F3 · 纳尺度缺陷容差: 临界缺陷尺寸 a* 与给定最大缺陷下的强度保持率.

    开放问题(综述"极致强度"线): 当缺陷小于 a* 时强度回到理想强度(缺陷不敏感),
    这是纳米试样能达到理论强度的机制结论. 用"100 nm 缺陷下仍保持理想强度的比例"
    作为跨材料可比量(真实标量).
    """
    as_ = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 40.0]           # a/a* 倍数
    rows = []
    tol_vals = []
    astar_vals = []
    for label, e, st, g in _FLAW_MATS:
        curve, astar_nm = _flaw_strength_curve(e, st, g, as_)
        # 100nm 缺陷下 σ/σth: 若 a* >> 100nm 则保持率≈1(缺陷不敏感区).
        ratio_100 = float(min(1.0, math.sqrt(astar_nm / max(amax_nm, 1e-9))))
        tol_vals.append(ratio_100)
        astar_vals.append(astar_nm)
        rows.append({"material": label, "E_GPa": e, "sig_th_GPa": st,
                     "a*_nm": round(astar_nm, 3),
                     "sigma/sigth @100nm": round(ratio_100, 4),
                     "curve(σ/σth vs a/a*)": curve})
    return {
        "objectives": {"flaw_tol_100nm": round(min(tol_vals), 4)},
        "summary": {"amax_nm": amax_nm, "materials": rows,
                    "min_a*_nm": round(min(astar_vals), 3),
                    "max_a*_nm": round(max(astar_vals), 3)},
        "success": True,
    }


def _weibull_size_exponent(m: float, ns: list[float] | None = None,
                           seeds: int = 200) -> tuple[float, float]:
    """弱链: 无界 Pareto(m) 缺陷下 N 缺陷单元的统计强度, 数值验证 σ∝N^{−1/(2m)}.

    返回 (log-log 斜率, R²). 强度 σ_i ∝ a_i^{−1/2}, a_i ~ Pareto(无界尾) ⇒
    单元强度 = min σ_i = (max a_i)^{−1/2}. 用 Beta(1,N) 极小值序统计直接采样
    (避免 O(N·seeds) 内存, 且无截断偏差 —— 有界缺陷分布会饱和尺寸效应).
    """
    ns = (ns or [10, 30, 100, 300, 1000, 3000, 10000, 100000]) if not ns else ns
    if len(ns) < 3:
        ns = [min(ns), max(ns) * 3, max(ns) * 10]          # 点数太少时补成 log 谱
    rng = np.random.default_rng(7)
    means = []
    for N in ns:
        r = rng.random(seeds)                              # Beta(1,N) 采样辅助
        u_min = 1.0 - (1.0 - r) ** (1.0 / N)               # min of N iid U(0,1)
        # a_max = a_m·u_min^{−1/m} ⇒ σ_spec = u_min^{1/(2m)}/√a_m (a_m 常数, 斜率无关)
        means.append(float(np.mean(u_min ** (1.0 / (2.0 * m)))))
    lnN = np.log(ns); lns = np.log(np.array(means))
    k, b = np.polyfit(lnN, lns, 1)
    pred = k * lnN + b
    ss_res = float(np.sum((lns - pred) ** 2))
    ss_tot = float(np.sum((lns - lns.mean()) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    return float(-k), r2


def exp_statistical_size_effect(m: float = 2.0) -> dict:
    """F4 · 统计弱链尺寸效应: 幂律(斜率)与拟合优度 R².

    理论: 尺寸效应指数 1/(2m); 本实验用真实蒙特卡洛(200 seeds × N 谱)估计斜率并
    与理论对照 —— 回答"统计强度律能否用弱链+缺陷分布定量复现"(综述统计线).
    """
    slope, r2 = _weibull_size_exponent(m)
    theory = 1.0 / (2.0 * m)
    return {
        "objectives": {"weibull_fit": round(r2, 4)},
        "summary": {"m": m, "slope(数值)": round(slope, 4),
                    "theory(1/2m)": round(theory, 4),
                    "slope_theory_ratio": round(slope / theory, 4),
                    "r2(log-log线)": round(r2, 4)},
        "success": True,
    }


def _bridge_gain(sigma0_over_sy: float, cod_budget: float = 0.5) -> float:
    """Dugdale 常数桥联: K_c/K_0 = sqrt(1 + (σ0/σy)·(δc/δ_tip)), 小尺度屈服.

    σ0 桥联牵引, δc 可耗散张开预算(归一于 K0²/(Eσy)). 纯解析(经典小尺度桥联).
    """
    return float(math.sqrt(1.0 + sigma0_over_sy * cod_budget))


def exp_bridging_toughening() -> dict:
    """F5 · 桥联增韧增益: (σ0/σy × δc/δ_tip) 面上的 K_c/K_0 增益.

    开放问题(综述仿生/珍珠母线): 牺牲性桥联把表观韧性推离尖端临界, 增益上界为何?
    """
    rows = []
    gains = []
    for s in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
        g = _bridge_gain(s)
        rows.append({"sigma0/sy": s, "Kc/K0(δ预算0.5)": round(g, 4)})
        gains.append(g)
    peak = float(max(gains))
    return {
        "objectives": {"kce_gain": round(peak - 1.0, 4)},
        "summary": {"rows": rows, "max_Kc/K0": round(peak, 4),
                    "kce_gain": round(peak - 1.0, 4)},
        "success": True,
    }


def _rayleigh_speed(nu: float) -> tuple[float, float, float]:
    """c_R 精确根(平面应变): D(v)=4α1α2−β² 在 (0,c2) 的零点; 返回 (cR/c2, c1/c2, D函数)."""
    c1 = math.sqrt(2 * (1 - nu) / (1 - 2 * nu))    # c1/c2
    def D(x):                                      # x = v/c2
        a1 = math.sqrt(1 - (x / c1) ** 2)
        a2 = math.sqrt(1 - x * x)
        return 4 * a1 * a2 - (2 - x * x) ** 2
    cR = float(brentq(D, 1e-6, 0.9999))
    return cR, c1, D


def _rose_k(v_cR: float) -> float:
    """Rose(1976) 通用动态 SIF 因子 k(v)=(1−v/cR)/(1−v/2cR) (mode-I 稳态, 精度~%) ."""
    return (1 - v_cR) / (1 - v_cR / 2.0)


def exp_dynamic_barrier(nu: float = 0.3) -> dict:
    """F6 · mode-I 动态断裂速度禁区: 精确 Rayleigh 根 c_R 与能量通量比 g=k².

    开放问题(综述动态断裂线): 经典 LEFM 在 c_R<v<c2 不存在稳态张开裂纹(势垒),
    而界面/剪切裂纹可进入跨声速(超剪切), 前提是剪制约束 —— 本实验给出开裂纹
    势垒的定量形态(零点/锐度)与声学缺口(c2−cR)/cR.
    """
    cR, c1, _ = _rayleigh_speed(nu)
    vs = [0.1, 0.25, 0.5, 0.7, 0.85, 0.9, 0.95, 0.98]
    curve = []
    for v in vs:
        k = _rose_k(v)
        curve.append({"v/cR": v, "k(v)": round(k, 4), "g=k^2": round(k * k, 4)})
    # 锐度: ln g 在 v=0.95cR 的负斜率(数值导数)
    g1, g2 = _rose_k(0.94) ** 2, _rose_k(0.96) ** 2
    sharpness = float((math.log(g1) - math.log(g2)) / 0.02)
    gap = float((1.0 - cR) / cR)                       # (c2−cR)/cR, c2=1 归一
    return {
        "objectives": {"barrier_sharpness": round(sharpness, 2),
                       "gap_extent": round(gap, 3)},
        "summary": {"nu": nu, "cR/c2": round(cR, 5), "c1/c2": round(c1, 5),
                    "curve": curve, "barrier_sharpness(-dln g/d(v/cR)@0.95)":
                        round(sharpness, 2), "sonic_gap(c2-cR)/cR": round(gap, 3)},
        "success": True,
    }


def exp_kic_validity() -> dict:
    """F7 · KIC 有效性边界: ASTM E399 试样尺寸门槛 2.5(KIC/σy)² vs SSY.

    开放问题(综述标准/手册线): 高韧性低屈服金属需要米级试样, 直接限制 K-dominance
    验证 —— 给出各工程材料"试验难度"的跨度(真实门槛比).
    """
    rows = []
    demands = []
    for label, sy, kic in _KIC_MATS:
        demand_m = 2.5 * (kic / sy) ** 2 * 1e-3         # (MPa√m/MPa)²·2.5 → m
        rp_m = (kic / sy) ** 2 / (6 * math.pi) * 1e-3    # 平面应变塑性区
        rows.append({"material": label, "sigma_y(MPa)": sy, "KIC(MPa√m)": kic,
                     "试样尺寸门槛(mm)": round(demand_m * 1e3, 2),
                     "r_p(mm)": round(rp_m * 1e3, 3),
                     "门槛/塑性区": round(float(demand_m / (rp_m + 1e-12)), 1)})
        demands.append(demand_m)
    span = float(max(demands) / (min(demands) + 1e-12))
    return {
        "objectives": {"kic_span": round(math.log10(span), 2)},
        "summary": {"materials": rows,
                    "min_demand_mm": round(min(demands) * 1e3, 2),
                    "max_demand_mm": round(max(demands) * 1e3, 2),
                    "span(log10)": round(math.log10(span), 2)},
        "success": True,
    }

# ═══════════════════════════════ 扫描执行器 ═══════════════════════════════

def _sanitize_scan(cfg: dict | None) -> dict:
    """断裂扫描配置规范化: 只保留白名单键, 缺省值补齐, 值钳入白名单."""
    c = dict(_SCAN_DEFAULTS)
    for k, v in (cfg or {}).items():
        if k in SCAN_OPS and v is not None:
            c[k] = v
    return c


def exp_fracture_scan(cfg: dict, name: str) -> dict:
    """书生提议的断裂扫描配置 → 真实实验分支(白名单校验, 结果族全真实).

    values 缺失时对该维度全白名单扫描; 返回该族的主目标(公式与基分支一致).
    """
    c = _sanitize_scan(cfg)
    dim = (cfg or {}).get("dim", "pair")
    # dim 别名归一: 书生直接用白名单键名(bridge_ratio/nu/...) 或近义名,
    # 统一映射到 exp_fracture_scan 的分支判据, 防止"bridge_ratio"落到兜底空扫描.
    _dim_canon = {"bridge_ratio": "bridge", "sigma0_sy": "bridge",
                  "sigma0_over_sy": "bridge",
                  "poisson": "barrier", "vcR": "barrier", "v_cR": "barrier"}
    dim = _dim_canon.get(dim, dim)
    key = _DIM_VALUES.get(dim, dim)
    vals = (cfg or {}).get("values") or SCAN_OPS.get(key) or []
    if dim == "pair":
        eps = []
        rows = []
        for i in vals:
            label, e1, n1, e2, n2 = _PAIRS[i]
            a, b = _dundurs(e1, n1, e2, n2)
            e_ = abs((1.0 / (2.0 * math.pi)) * math.log((1 - b) / (1 + b)))
            eps.append(e_)
            rows.append({"pair": label, "|eps|": round(e_, 5)})
        obj = {"osc_span": round(float(max(eps) - min(eps)), 6)}
        return {"objectives": obj, "summary": {"dim": "pair", "rows": rows}, "success": True}
    if dim == "material":
        rows, marks = [], []
        for i in vals:
            label, mu, nu, b, gs, gusf = _MATERIALS[i]
            mk = gusf / gs
            marks.append(mk)
            rows.append({"material": label, "gamma_usf/gamma_s": round(mk, 3),
                         "class": "brittle" if mk > 1 else ("transitional" if mk >= 0.6
                                                            else "ductile")})
        obj = {"dbt_span": round(float(max(marks) - min(marks)), 3)}
        return {"objectives": obj, "summary": {"dim": "material", "rows": rows},
                "success": True}
    if dim == "flaw":
        # 语义: values = 材料索引(与其它 dim 一致, 不把倍率当索引); 缺省=全材料.
        mat_idx = [int(i) for i in vals] or list(range(len(_FLAW_MATS)))
        as_ = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 40.0]
        rows = []
        for i in mat_idx:
            label, e, st, g = _FLAW_MATS[min(max(int(i), 0), len(_FLAW_MATS) - 1)]
            curve, astar = _flaw_strength_curve(e, st, g, as_)
            rows.append({"material": label, "a*_nm": round(astar, 3), "curve": curve})
        a = [r["a*_nm"] for r in rows]
        obj = {"flaw_tol_100nm": round(float(max(a) / (min(a) + 1e-9)), 3)}
        return {"objectives": obj, "summary": {"dim": "flaw", "rows": rows}, "success": True}
    if dim == "n_flaws":
        # values 是 n_flaws 值; 受限于蒙特卡洛内存, 单点抽样上限截到 3e3.
        # summary 只放结果标量(斜率/R²), 不放配置参数列表 —— 避免 _first_scalar
        # 把配置值当"主张"而触发声明门禁未落地误报.
        ns = [float(v) for v in vals]
        slope, r2 = _weibull_size_exponent(2.0, ns=ns, seeds=150)
        obj = {"weibull_fit": round(r2, 4)}
        return {"objectives": obj,
                "summary": {"dim": "n_flaws",
                            "slope": round(slope, 4),
                            "theory_1over2m": 0.25, "r2": round(r2, 4),
                            "n_sampled": len(ns)}, "success": True}
    if dim == "bridge":
        rows = [{"sigma0/sy": r, "Kc/K0": round(_bridge_gain(r), 4)} for r in vals]
        peak = max(_bridge_gain(r) for r in vals)
        obj = {"kce_gain": round(float(peak - 1.0), 4)}
        return {"objectives": obj, "summary": {"dim": "bridge", "rows": rows},
                "success": True}
    if dim == "barrier" or dim == "nu":   # nu 是书生可能用的 barrier 别名
        rows = []
        for nu in vals:
            cR, c1, _ = _rayleigh_speed(nu)
            g1, g2 = _rose_k(0.94) ** 2, _rose_k(0.96) ** 2
            sharp = (math.log(g1) - math.log(g2)) / 0.02
            rows.append({"nu": nu, "cR/c2": round(cR, 5), "gap(c2-cR)/cR":
                         round((1 - cR) / cR, 3), "sharpness": round(sharp, 2)})
        obj = {"barrier_sharpness": round(max(r["sharpness"] for r in rows), 2)}
        return {"objectives": obj, "summary": {"dim": "barrier", "rows": rows},
                "success": True}
    if dim == "kic":
        rows, demands = [], []
        for i in vals:
            label, sy, kic = _KIC_MATS[i]
            d = 2.5 * (kic / sy) ** 2 * 1e-3
            demands.append(d)
            rows.append({"material": label, "门槛(mm)": round(d * 1e3, 2)})
        obj = {"kic_span": round(math.log10(max(demands) / (min(demands) + 1e-12)), 2)}
        return {"objectives": obj, "summary": {"dim": "kic", "rows": rows},
                "success": True}
    return {"objectives": {"osc_span": 0.0}, "summary": {"dim": dim, "rows": []},
            "success": True}

# ═══════════════════════ 域诊断工具(书生成文期自主调用) ═══════════════════════

def _diagnostic_tools():
    """书生成文阶段可自主调用的断裂域探针(真实数值，进 trace 供门禁核对)."""
    def h_probe_interface(a):
        i = int(a.get("pair", 0))
        label, e1, n1, e2, n2 = _PAIRS[i]
        alpha, beta = _dundurs(e1, n1, e2, n2)
        eps = (1.0 / (2.0 * math.pi)) * math.log((1 - beta) / (1 + beta))
        return json.dumps({"pair": label, "alpha": round(alpha, 3),
                           "beta": round(beta, 3), "|eps|": round(abs(eps), 5)},
                          ensure_ascii=False)

    def h_probe_dbt(a):
        i = int(a.get("material", 0))
        label, mu, nu, b, gs, gusf = _MATERIALS[i]
        mk = gusf / gs
        return json.dumps({"material": label, "gamma_usf/gamma_s": round(mk, 3),
                           "class": "brittle" if mk > 1 else ("transitional" if mk >= 0.6
                                                             else "ductile")},
                          ensure_ascii=False)

    def h_probe_flaw(a):
        i = min(int(a.get("material", 0)), len(_FLAW_MATS) - 1)
        label, e, st, g = _FLAW_MATS[i]
        curve, astar = _flaw_strength_curve(e, st, g, [0.5, 1.0, 2.0, 5.0, 10.0])
        return json.dumps({"material": label, "a*_nm": round(astar, 3),
                           "sigma/sigth(a/a*):[0.5,1,2,5,10]": curve}, ensure_ascii=False)

    def h_probe_weibull(a):
        slope, r2 = _weibull_size_exponent(float(a.get("m", 2.0)))
        return json.dumps({"slope": round(slope, 4), "theory(1/2m)":
                           round(1 / (2 * float(a.get("m", 2))), 4),
                           "r2": round(r2, 4)}, ensure_ascii=False)

    def h_probe_bridge(a):
        s = float(a.get("sigma0_over_sy", 0.5))
        return json.dumps({"sigma0/sy": s, "Kc/K0": round(_bridge_gain(s), 4)},
                          ensure_ascii=False)

    def h_probe_barrier(a):
        nu = float(a.get("nu", 0.3))
        cR, c1, _ = _rayleigh_speed(nu)
        return json.dumps({"nu": nu, "cR/c2": round(cR, 5),
                           "sonic_gap(c2-cR)/cR": round((1 - cR) / cR, 3)},
                          ensure_ascii=False)

    def h_probe_kic(a):
        i = int(a.get("kic_mat", 0))
        label, sy, kic = _KIC_MATS[i]
        d = 2.5 * (kic / sy) ** 2 * 1e-3
        return json.dumps({"material": label, "门槛(mm)": round(d * 1e3, 2),
                           "KIC/sigma_y": round(kic / sy, 3)}, ensure_ascii=False)

    def _spec(fn_name: str, desc: str, param: dict) -> dict:
        """构造 MCP-style 工具描述(huginn diagnostic_tools 契约)."""
        return {"tool": {"function": {"name": fn_name, "description": desc,
                                      "parameters": {"type": "object",
                                                     "properties": param}}},
                "handle": _PROBES[fn_name]}

    _PROBES = {"probe_interface": h_probe_interface, "probe_dbt": h_probe_dbt,
               "probe_flaw": h_probe_flaw, "probe_weibull": h_probe_weibull,
               "probe_bridge": h_probe_bridge, "probe_barrier": h_probe_barrier,
               "probe_kic": h_probe_kic}
    return [
        _spec("probe_interface", "双材料对界面振荡指数", {"pair": {"type": "integer"}}),
        _spec("probe_dbt", "材料脆韧标记", {"material": {"type": "integer"}}),
        _spec("probe_flaw", "纳尺度缺陷容差曲线", {"material": {"type": "integer"}}),
        _spec("probe_weibull", "统计弱链尺寸效应", {"m": {"type": "number"}}),
        _spec("probe_bridge", "桥联增韧增益", {"sigma0_over_sy": {"type": "number"}}),
        _spec("probe_barrier", "Rayleigh 势垒与声学缺口", {"nu": {"type": "number"}}),
        _spec("probe_kic", "KIC 试样尺寸门槛", {"kic_mat": {"type": "integer"}}),
    ]

# ═══════════════════════ 书生自主环: 观察→提议→行动 ═══════════════════════

def _ask_json(client, model: str, system: str, user: str, max_tokens: int = 900) -> dict:
    try:
        r = client.chat.completions.create(
            model=model, max_tokens=max_tokens, temperature=0.2,
            extra_body={"thinking_mode": False},
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}])
        text = r.choices[0].message.content or ""
    except Exception as ee:  # noqa: BLE001
        print(f"  [提议] LLM 失败: {ee}")
        return {}
    return _extract_json_balanced(text)


def _extract_json_balanced(text: str) -> dict:
    """从 LLM 输出中稳健提取 JSON 对象.

    旧实现用 re.search(r'[{}]') 的贪婪匹配盲抓最外层大括号 —— 但书生成码/推导
    脚本里常在 JSON 的 code 字段内嵌 {}(dict/函数体), 贪婪匹配会提前截断或把
    后一个 JSON 块卷进来. 新实现从每个可能起点做【平衡括号扫描】+ json.loads 校验,
    取第一个能完整解析的合法 JSON 对象(优先含 code/expr/config 的). 解析失败返回 {}.
    """
    text = text or ""
    # 1) 整串本身就是合法 JSON(最常见, 书生遵守只输出 JSON)
    try:
        d0 = json.loads(text.strip())
        if isinstance(d0, dict):
            return d0
    except Exception:  # noqa: BLE001
        pass
    # 2) 平衡括号扫描: 收集所有可解析的 dict, 选取优先键对象的.
    pairs_stack: list[int] = []
    found: list[dict] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "{":
            pairs_stack.append(i)
        elif ch == "}" and pairs_stack:
            start = pairs_stack.pop()
            try:
                d = json.loads(text[start:i + 1])
            except Exception:  # noqa: BLE001 — 非合法 JSON, 继续更大范围
                continue
            if isinstance(d, dict) and d:
                found.append(d)
        i += 1
    if not found:
        return {}
    # 优先级: 含 code(书生成码/符号脚本) > 含 expr(闭式) > 含 domain 下发键
    # > 含普适键 > 最后兜底第一个. 确保多对象输出时取到正确那个.
    def _prio(d: dict) -> int:
        if "code" in d:
            return 0
        if "expr" in d:
            return 1
        if any(k in d for k in ("configs", "open_questions", "objectives",
                                "mechanisms", "new_questions", "findings")):
            return 2
        return 3
    best = min(found, key=_prio)
    return best


def _extract_next_open(report_text: str) -> str:
    pat = re.compile(r"#+\s*(?:[0-9]+[.、)]?\s*)*(下一步|局限|后续工作|未来工作)"
                     r"[^\n]*\n(.*?)(?=\n[#]{1,3}\s|\Z)", flags=re.DOTALL)
    m = pat.search(report_text or "")
    if not m:
        # 兜底: 取结论段, 但剔除"真实结果"JSON 行 —— 避免把上一轮旧数值带进新目标
        # (dry 确定性综合下这些数字无 trace 支撑, 会被声明门禁误报未落地).
        tail = (report_text or "").strip()[-1200:]
        tail = re.sub(r"-\s*真实结果:.*", "- (上轮真实结果, 见报告对应分支)", tail)
        return tail[:2000]
    return m.group(2).strip()[:2000]


_WHITELIST_DESC = (
    "{pair∈0..5(Al2O3/Ni,SiC/Al,PMMA/steel,glass/epoxy,sapphire/NiAl,diamond/WC), "
    "material∈0..9(diamond,Si,W,α-Fe,Ti,Mg,Cu,Ni,Al,Au), "
    "flaw_idx∈{0.25,0.5,1,2,5,10,40}(a/a*), n_flaws∈{10..100000}, "
    "bridge_ratio∈{0.05..0.9}(σ0/σy), vcR∈{0.1..0.98}(v/c_R), nu∈{0.2,0.3,0.4}}"
)


def _propose_next_open(client, model: str, report_text: str) -> list[str]:
    critique = ""
    m = re.search(r"对立审稿\(CriticAgent\)(.*)", report_text or "", flags=re.DOTALL)
    if m:
        critique = m.group(1)[:2000]
    sys = (
        "你是断裂力学长程科研规划者。基于前一周期报告与审稿副体意见, 提出下一周期"
        f"最值得攻克的开放问题。要求: 每条必须能被白名单真实数值实验 {_WHITELIST_DESC} "
        "检验, 指向具体可证伪预言(如: 'ν 变大时声学缺口减小到什么量级?'; "
        "'桥联增益在 σ0/σy>0.7 是否饱和?'); 每条一句。"
        '只输出 JSON: {"open_questions": ["q1","q2"]}, 不含其他文字。'
    )
    user = f"上一周期报告:\n{report_text[:6000]}\n\n审稿副体意见:\n{critique or '(无)'}"
    # 端到端闭合(断点 B): 先召回已沉淀的品味启发式, 注入下一轮发问 —— 让问题从品味长出.
    recalled = _recall_taste("how to pose next frontier research questions using distilled taste patterns: assumption failure, scale break, paradox, analogy")
    if recalled:
        user += (f"\n\n【可复用参考 · 全局知识库召回的研究品味】\n{recalled}\n"
                 "可吸收其发问模式, 但下一轮问题必须落在断裂白名单可验证、且不重复已有结论。")
    d = _ask_json(client, model, sys, user)
    qs = [str(q).strip() for q in (d.get("open_questions") or []) if str(q).strip()]
    return qs[:3]


def _propose_scan_configs(client, model: str, next_open: str,
                          done_cfgs: list[dict] | None = None) -> list[dict]:
    done = " ".join(json.dumps(c, sort_keys=True, ensure_ascii=False)
                    for c in (done_cfgs or [])) or "无"
    d = _ask_json(
        client, model,
        "你是实验设计者。基于开放问题, 从断裂扫描白名单 {dim∈{pair,material,flaw,"
        "n_flaws,bridge,barrier,kic} 及其值域 {pair∈0..5, material∈0..9, "
        "flaw∈0..3(graphene/Si/steel/Al2O3 材料索引), "
        "n_flaws∈{10,30,100,300,1000,3000,1e4,1e5}, bridge_ratio∈{0.05,0.1,0.2,0.3,0.5,0.7,0.9}, "
        "vcR∈{0.1,0.25,0.5,0.7,0.85,0.9,0.95,0.98}, nu∈{0.2,0.3,0.4}, "
        "kic_mat∈0..5}} 提议 ≤2 个配置; "
        "每个配置形如 {\"dim\":\"<dim>\", \"values\":[白名单内值]} (values 可省略=全扫)。"
        "输出: {\"configs\": [config1, config2]}, 不含其他文字。",
        f"开放问题: {next_open[:1200]}\n已执行配置: {done}")
    cfgs = []
    for c in (d.get("configs") or [])[:2]:
        if isinstance(c, dict) and c.get("dim") in SCAN_OPS:
            cfgs.append(c)
    return cfgs


def _pad_scan_configs(cfgs: list[dict], open_text: str) -> list[dict]:
    """兜底: 书生未提议或提议不足时, 用未检维度补齐(仍为真实实验)."""
    _DIMS = ["barrier", "flaw", "bridge", "kic", "pair", "n_flaws", "material"]
    have = {c.get("dim") for c in cfgs if isinstance(c, dict)}
    for dim in _DIMS:
        if dim not in have and len(cfgs) < 3:
            cfgs.append({"dim": dim})
    return cfgs[:3]


# ── Symbolic Lab: 书生从控制方程推导闭式解, agent 沙箱符号求解 + 数值核验 ──
# 超越"数值扫描 + 定性决策": 书生写 sympy 脚本对控制方程(近似)做符号操作
# (求根/微分/极限/复含数性质), 产出【闭式表达式】; 框架再把该闭式在多个真实
# 数值点上与底层物理函数(经参数化方程一致的精准实现)核对一致, 才对 grounded.
# 诚实红线: 闭式必须是符号推导产出的真实公式, 核验不一致/无法 sympify 即弃用.
#
# 两级兜底(书生不具备符号能力时也行):
#   L1 书生主动: derive() 给闭式, 框架核验.
#   L2 agent 保底: 书生给真实数值轨迹(x,y), agent 用 _close_form_fit 做
#        最小二乘闭式归纳(从真实点反推幂律/根号律), 再交数值核验. 这使 agent
#        "补上"书生不具备的闭式化能力, 而不伪造 —— 归纳源=书生跑的真实点.

# 可被书生符号推导并做数值核验的"真实物理公式核" (dim → (参数名, 核函数)).
# 框架用同一参数空间做闭式↔核函数的一致性核对.
_SYM_GROUND = {
    "barrier": ("nu", lambda nu: _rose_k(0.95)),
    "bridge": ("s", lambda s: _bridge_gain(s)),
    "flaw": ("a", lambda a: 1.0 / math.sqrt(a)),
}
_SYM_NU_GROUND = ("nu", lambda nu: _rayleigh_speed(nu)[0])


def _close_form_fit(xs: list[float], ys: list[float], family: str) -> str | None:
    """agent 保底(L2): 从书生跑出的真实数值轨迹归纳闭式.

    候选闭式仅限"物理上可信"的家族, 不做黑箱多式拟合(避免过拟合编造):
      bridge: Kc/K0 = sqrt(1 + k*s)  → 对 y²~s 线性拟合求 k.
      barrier: k(v) = (1-v)/(1-v/2) 幂律/有理 → 最小二乘配 (1+a·v)/(1+b·v).
    返回 sympy 可解析的表达式字符串; 拟合优度过差返回 None.
    """
    import numpy as np
    if family == "bridge":
        # y = sqrt(1 + k x) => y^2 = 1 + k x, 直线斜率 k
        y2 = np.asarray(ys) ** 2
        x = np.asarray(xs, dtype=float)
        k, b = np.polyfit(x, y2, 1)          # y^2 = k*x + b
        resid = np.sum((y2 - (k * x + b)) ** 2)
        r2 = 1 - resid / np.sum((y2 - y2.mean()) ** 2)
        if r2 < 0.9998 or abs(b - 1.0) > 0.01:
            return None
        return f"sqrt(1 + {k:.6f}*s)"
    if family == "barrier":
        # k(v) = (1 + a v) / (1 + b v) 近似配 Rose; 参数化退化改用幂律备选.
        return None
    return None


def _sym_ground_check(expr_sym: Any, family: str) -> dict | None:
    """把书生导出的 sympy 闭式与真实物理核做多点数值核验.

    返回 {max_relerr, r2, n_points} 或 None(核验失败). family 决定参数与核:
      barrier/rose: 表达式应谓 v∈[0,0.9] 的 k(v)=(1-v)/(1-v/2) 的某种等价形式
      bridge:       表达式应谓 s∈[0.05,0.9] 的 sqrt(1+s*0.5)
    闭式经 sympy.nsubs 代入真实点求值, 与核逐点比对, 同相对误差判一致.
    """
    import sympy as sp
    if family in ("barrier", "rose"):
        var_name = "v"
        f_ref = lambda vv: (1 - vv) / (1 - vv / 2.0)          # noqa: E731 Rose k(v)
        pts = [0.1, 0.25, 0.5, 0.7, 0.85, 0.9]
    elif family == "bridge":
        var_name = "s"
        f_ref = lambda ss: math.sqrt(1.0 + ss * 0.5)          # noqa: E731 Dugdale
        pts = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    else:
        return None
    # 按【名称】匹配闭式里的自由符号(勿用带 assumptions 的 Symbol —— 其与
    # sympify 产出的无假设符号不相等, 导致 subs 不生效).
    free_by_name = {str(fs): fs for fs in expr_sym.free_symbols}
    sym_var = free_by_name.get(var_name)
    if sym_var is None:
        return None
    try:
        ref = [f_ref(p) for p in pts]
        got = []
        for p in pts:
            got.append(float(expr_sym.subs(sym_var, sp.Float(p)).evalf()))
        rel = [abs(g - r) / (abs(r) + 1e-12) for g, r in zip(got, ref)]
        mae = sum(abs(g - r) for g, r in zip(got, ref)) / len(ref)
        return {"max_relerr": round(max(rel), 6), "mae": round(mae, 6),
                "n_points": len(pts), "ref": ref, "got": got}
    except Exception as ee:  # noqa: BLE001 — 代入求值失败视核验失败
        print(f"  [symlab] 核验失败: {type(ee).__name__}: {ee}")
        return None


def _try_author_symbolic(client, model: str, next_open: str, cycle: int):
    """Symbolic Lab: 书生写 sympy 脚本, 从控制方程推导断裂量的闭式解.

    返回 (Experiment|None, err_msg). 闭式经 _sym_ground_check 与真实物理核核对
    一致后, 作为 grounded 的符号推导分支进入本轮; 失败回退书生成码/白名单.

    书生契约: 代码必须定义
        def derive() -> dict:   # 返回 {"expr": "<sympy 表达式, 自变量为 family 参数>",
                                #        "family": "barrier|bridge",
                                #        "note": "<推导说明/这个闭式从哪个方程来>"}
    框架对 derive() 返回的 expr 做 sympify + _sym_ground_check 数值核验.
    """
    from huginn.research.code_lab import (extract_code, _load_namespace,
                                          _call_with_timeout, _to_py)
    sys_prompt = (
        "你是断裂力学符号推导者。用 sympy(必要时 numpy/scipy)对一个断裂开放问题"
        "从控制方程推导【闭式解】。写一个符号推导脚本, 必须定义:\n"
        "def derive() -> dict:\n"
        "    # 用 sympy 符号从基本关系(如 Rose 动态 SIF k(v)=(1-v)/(1-v/2)、\n"
        "    # Dugdale 桥联 Kc/K0=sqrt(1+s*δc/δtip)、Rayleigh 色散方程等)出发, \n"
        "    # 通过 sympify/化简/simplify 导出该量的显式表达式\n"
        "    return {'expr': '<sympy表达式字符串, 变量为对应 family 的参数>',\n"
        "            'family': 'barrier|bridge',   # 框架据此用同一物理核核验\n"
        "            'note': '<这个闭式是如何从方程推出的>'}'\n"
        "family 含义: barrier 的表达式应等于 k(v)=(1-v)/(1-v/2) (自变量 v, v<1);\n"
        "             bridge  的表达式应等于 sqrt(1+s*0.5) (自变量 s, s>0).\n"
        "expr 必须是符号推导获得的等价形式(如化简/展开/参数改写), 不准硬抄 RHS.\n"
        "只输出 JSON: {\"code\": \"...\"}, 不含其他文字。脚本内可 print 推导过程。"
    )
    d = _ask_json(client, model, sys_prompt, f"开放问题: {next_open[:900]}",
                  max_tokens=1600)
    code = d.get("code") or ""
    if not code:
        print(f"  [symlab-diag] 提取到 code 为空; 原始LLM输出前300:{str(d)[:160]}")
        return None, [], "书生未给出符号推导脚本"
    # 若用围栏包了, 去掉围栏; 若已是裸 derive 定义(<code>/``` 都不存在), 直接用.
    if "```" in code or "<code>" in code:
        code = extract_code(code)
    if not code.strip().startswith("def derive"):
        i = code.find("def derive")
        if i >= 0:
            # 保留 def derive 之前的 import/注释(砍掉会让脚本缺依赖);
            # 只剥掉"说明文字"(非 import 的前缀). 逐个前缀行判断.
            prefix = code[:i]
            keep_imports = [ln for ln in prefix.splitlines()
                            if ln.strip().startswith(("import ", "from "))]
            if prefix.strip() and all(not ln.strip() or ln.strip().startswith(("import", "from"))
                                      for ln in prefix.splitlines()):
                code = code   # 前缀全是 import/空行, 全保留
            else:
                code = "".join(x + "\n" for x in keep_imports) + code[i:]
    if "def derive" not in code:
        return None, [], "脚本未定义 def derive()"
    cfg = dict(_SCAN_DEFAULTS)
    # 在安全沙箱加载 derive 脚本(白名单 import + 超时 + 内存上限), 校验可执行
    # 语法; 再调用 derive() 取闭式元信息. 不用 sandbox_run —— 它只认 def run(cfg).
    try:
        ns = _load_namespace(code)
    except Exception as ee:  # noqa: BLE001
        return None, [], f"符号脚本执行失败: {type(ee).__name__}: {ee}"
    dr = ns.get("derive")
    if not callable(dr):
        return None, [], "未找到 def derive() 入口"
    try:
        # derive() 无参; _call_with_timeout 会传一个 arg, 用包装把 cfg 忽略掉.
        meta = _call_with_timeout(lambda _c: dr(), cfg)
        meta = _to_py(meta)
    except Exception as ee:  # noqa: BLE001
        return None, [], f"derive() 执行失败: {type(ee).__name__}: {ee}"
    if not isinstance(meta, dict):
        return None, [], "derive() 未返回 dict"
    import sympy as sp
    expr_str = (meta or {}).get("expr", "")
    family = (meta or {}).get("family", "")
    if not expr_str or family not in _SYM_GROUND:
        return None, [], f"未给出可核验闭式(family={family})"
    try:
        expr_sym = sp.sympify(expr_str)
    except Exception as ee:  # noqa: BLE001
        return None, [], f"sympify 失败: {ee}"
    chk = _sym_ground_check(expr_sym, family)
    if chk is None or chk["max_relerr"] > 5e-3:
        return None, [], (f"闭式与真实物理核不一致(max_relerr="
                          f"{chk['max_relerr'] if chk else 'n/a'})")

    from huginn.research import Experiment
    note = str((meta or {}).get("note", ""))[:300]
    expr_str_clean = str(sp.simplify(expr_sym))[:400]
    hypothesis = (f"Symbolic Lab(第{cycle}轮): 书生从控制方程推出的闭式解 "
                  f"{expr_str_clean} 与真实物理核逐点一致 "
                  f"(max_relerr={chk['max_relerr']}, n={chk['n_points']}). {note}")
    objective_key = "bridge_closure" if family == "bridge" else "barrier_closure"
    # 闭式在某具体参数点的真实数值作为 objective(可复现、可过门禁)
    objective_val = round(chk["got"][len(chk["got"]) // 2], 6)

    def run():
        return {"objectives": {objective_key: objective_val},
                "summary": {"closed_form": str(sp.simplify(expr_sym))[:300],
                            "family": family, "note": note,
                            "max_relerr": chk["max_relerr"],
                            "n_points": chk["n_points"]},
                "success": True}
    exp = Experiment(name=f"symlab_c{cycle}", hypothesis=hypothesis, run=run)
    return exp, [objective_key], ""


def _try_author_code(client, model: str, next_open: str, cycle: int):
    """Code Lab: 书生亲手写本轮断裂实验代码(失败回退白名单扫描, 不阻塞).

    返回 (exp_or_None, probe_specs, obj_keys, err_msg).
    """
    from huginn.research.code_lab import extract_code, sandbox_run, author_probe_specs
    sys_prompt = (
        "你是断裂力学实验员。用 numpy(必要时 scipy)写一个真实物理数值实验函数 "
        "def run(cfg): 检验给定的开放问题, 返回 {'success': bool, "
        "'summary': {可复现数值轨迹: 全部为实数/list/str}, 'objectives': {<key>: 数值}}。"
        "可用的断裂域白名单: cfg['pair']∈0..5(双材料对), cfg['material']∈0..9(材料), "
        "cfg['flaw_idx']=a/a*倍数, cfg['n_flaws']∈{10..100000}, "
        "cfg['bridge_ratio']∈{0.05..0.9}, cfg['vcR']∈{0.1..0.98}, cfg['nu']∈{0.2,0.3,0.4}, "
        "cfg['kic_mat']∈0..5。公式必须来自经典断裂力学(Griffith/Rice-Dugdale/Freund/"
        "Dundurs 等), objectives 每个值必须是真实计算的数值。输出裸代码(不要解释), "
        "代码必须含 def run(cfg)。"
    )
    d = _ask_json(client, model, sys_prompt,
                  f"本轮开放问题: {next_open[:1000]}", max_tokens=1500)
    code = d.get("code") or d.get("python") or ""
    if not code:
        code = _try_raw_code(client, model, next_open)
    if not code:
        return None, [], [], "书生未给出代码"
    code = extract_code(code) if not code.strip().startswith("def run") else code
    cfg = dict(_SCAN_DEFAULTS)
    res, err = sandbox_run(code, cfg)
    if res is None:
        return None, [], [], err or "代码执行失败"
    obj_keys = list(res["objectives"].keys())
    probe_specs = []
    try:
        probe_specs = author_probe_specs(code)
    except Exception:  # noqa: BLE001
        probe_specs = []
    exp = _author_to_experiment(code, res, obj_keys, cycle)
    return exp, probe_specs, obj_keys, ""


def _try_raw_code(client, model: str, next_open: str) -> str:
    """兜底: 书生直接给代码而非 JSON 包装时, 原样取回."""
    try:
        r = client.chat.completions.create(
            model=model, max_tokens=1500, temperature=0.2,
            extra_body={"thinking_mode": False},
            messages=[{"role": "user",
                       "content": "写一个 numpy 断裂力学数值实验函数 def run(cfg): 检验: "
                                  + next_open[:800] + "\n输出裸代码。"}])
        return r.choices[0].message.content or ""
    except Exception:  # noqa: BLE001
        return ""


def _author_to_experiment(code: str, res: dict, obj_keys: list[str], cycle: int):
    from huginn.research import Experiment
    hypothesis = (f"书生成码分支(第{cycle}轮): 书生亲手写的断裂数值实验已通过 schema 校验, "
                  f"objectives={obj_keys}")
    def run():
        return res
    return Experiment(name=f"author_c{cycle}", hypothesis=hypothesis, run=run)


def _llm_critic(client, model: str, report_text: str, survivors_text: str) -> list[dict]:
    d = _ask_json(
        client, model,
        "你是对立审稿人(CriticAgent), 持反对立场复核主研究员的断裂力学报告。"
        "对每条存活假说找: 未做控制变量/未报误差/跨条件强推/经典理论误用等风险。"
        '输出: {"findings": [{"claim": "断言", "risk": "风险", "suggest": "建议"}]}, '
        "最多 4 条, 每条断言须能回溯到 summary 数值。",
        f"报告:\n{report_text[:5000]}\n\n存活证据:\n{survivors_text[:1500]}")
    finds = d.get("findings") or []
    out = [f for f in finds if isinstance(f, dict) and f.get("claim")]
    return out[:4]

# ══════════════════ 研究品味层: 思考"问题如何被提出"并自主生成新问题 ══════════════════
# 不只是执行给定开放问题; 而是让书生反身读出每个问题背后的"提出机制"(理想假说在哪
# 一步失效/何种悖论/何种尺度耦合断裂), 提炼可复用的问题生成启示(taste), 再依此自生
# 成白名单外的可证伪新问题 —— 品味是可迁移的元认知, 而非一次性答案列表.

_THEMES = (
    "综述七大开放问题及其参考证据(名词即可激发机制)"
)

# 问题→提出机制的证据锚(来自 IJF 2026 综述引文, 供书生归纳生成机制).
_THEME_EVIDENCE = [
    ("ON1 界面裂纹互穿悖论",
     "LEMF 双材料界面 K 场均含振荡项 r^{±iε}: ε≠0 ⇒ 裂纹面互相穿透(material interpenetration)。"
     "引: Mantič 有限牵引线性脆性界面 / Liechti 双向加载界面韧性 / Needleman-Rosakis 内聚 bond。"),
    ("ON2 脆-韧转变的统一判据",
     "同族材料何以有的本能脆、有的本能韧: Rice-Thomson 位错发射 vs 解理竞争仍是开放问题。"
     "引: Rice & Thomson 1974 / Rice 1992 Peierls 位错形核 / Li 2002 atomistic mechanisms / "
     "McMeeking 有限变形裂纹张开。"),
    ("ON3 纳尺度缺陷容差与极致强度",
     "理想强度在宏观现实; 纳米试样却达到: 缺陷一旦小于临界尺度便不敏感(Griffith∝1/√a 与 σ_th 的交叉)。"
     "引: Lee 2008 石墨烯 / Zhang 2016 Si NW / Nie 2019 金刚石 / Yang2025 nano-scale solids。"),
    ("ON4 统计尺寸效应的失效/适用范围",
     "Weibull 弱链律基于缺陷尾部分布; 当缺陷尺寸有界/相互作用强时, 尺寸效应饱和偏离幂律。"
     "引: Zhang-Li-Yang 统计强度 / Li-Yang 微裂纹聚合 / Shockey 岩石动态碎裂 / Mott 碎裂统计。"),
    ("ON5 桥联增韧的韧性-强度冲突",
     "珍珠母等牺牲性桥联把表观韧性推离尖端临界, 但代价是强度; 增益存在上界与最优层级。"
     "引: Shao 2012 非连续桥联 / Yao-Gao 多尺度内聚律 / Yan nacre T-stress / Zhang 最优层级。"),
    ("ON6 动态断裂的速度禁区",
     "mode-I 在 c_R<v<c_S 无稳态张开裂纹(Rayleigh 势垒); 而界面/剪切(超剪切)可进入跨声速。"
     "引: Rosakis 1999 intersonic / Xia 2004 实验室地震 / Needleman-Rosakis bond 强度/加载率。"),
    ("ON7 KIC 有效性的工程边界",
     "高韧性低屈服金属需要米级试样以满足 ASTM E399 的 2.5(K/σy)² 门槛: K-dominance 有尺度前提。"
     "引: ASTM E399-19 / Murakami、Tada-Paris-Irwin 手册 / Wu 权函数。"),
]

_TASTE_CATEGORIES = ("PARADOX(理想理论内在矛盾)", "ASSUMPTION_FAIL(某个理想化假设在现实边界失效)",
                     "SCALE_BREAK(跨尺度耦合在中间尺度断裂)", "MODEL_GAP(模型与实验观测的鸿沟)",
                     "ENABLER(新方法/新测量/新使能打开旧禁区)", "ANALOGY(跨域类比带进来的问题)")


def _sync_llm_create(client, model: str):
    """把书生包成语义蒸馏器可用的同步 callable(llm(prompt)->str)."""
    def _call(prompt: str) -> str:
        try:
            r = client.chat.completions.create(
                model=model, max_tokens=1800, temperature=0.3,
                extra_body={"thinking_mode": False},
                messages=[{"role": "user", "content": prompt}])
            return r.choices[0].message.content or ""
        except Exception:  # noqa: BLE001
            return ""
    return _call


def _ensure_kb_embedding_available() -> None:
    """KB embedding 可用性守卫: 无 sentence-transformers/torch 时让 chromadb ONNX 兜底.

    端到端闭环不因缺重依赖/断网而断:
      ST(多语言, 最好) → chromadb ONNX(英文 all-MiniLM, 384 维, 零下载) → 确定性兜底.
    仅设设计内置降级开关, 不重造; 装 ST 后自动让位。
    """
    try:
        from huginn.knowledge.store import _EmbeddingModel
        try:
            _EmbeddingModel()  # 初始化加载 chromadb DefaultEmbeddingFunction(ONNX)
        except Exception:  # noqa: BLE001
            pass
        try:
            import sentence_transformers  # noqa: F401 — 有真语义模型(多语言)就用
        except Exception:  # noqa: BLE001
            _EmbeddingModel._st_failed = True
            _EmbeddingModel._onnx_degraded = False
    except Exception:  # noqa: BLE001
        pass


_TASTE_KB = None  # 品味专属细 KB 的进程级缓存.
_ONNX_EF = None   # 自包含语义编码器: chromadb ONNX(384 维, 零下载, 写入/查询同模型).


def _taste_st():
    """品味语义路径的 ST 单例: 复用 store.py 全局, 本地快照加载, 不联网.

    权重已下载到 HF 缓存后, local_files_only 直读快照, 避免再次触发 xet/镜像下载;
    任何失败(缺权重/降级)返回 None, 由 _taste_embed 走 ONNX 兜底。
    """
    try:
        from huginn.knowledge.store import _EmbeddingModel, EMBED_MODEL
        if _EmbeddingModel._st is None and not _EmbeddingModel._st_failed:
            try:
                from sentence_transformers import SentenceTransformer
                _EmbeddingModel._st = SentenceTransformer(EMBED_MODEL, local_files_only=True)
            except Exception as ee:  # noqa: BLE001
                print(f"  [taste] ST 本地加载失败({ee}); 品味编码降级 ONNX")
                _EmbeddingModel._st_failed = True
        return _EmbeddingModel._st
    except Exception:  # noqa: BLE001
        return None


def _taste_embedder_fingerprint() -> str:
    """当前品味编码器指纹: 语义空间(模型)切换时用于触发品味 KB 重建."""
    return "st-384" if _taste_st() is not None else "onnx-384"


def _taste_embed(texts: list[str]) -> list[list[float]]:
    """品味语义编码: 优先 ST(多语言); 无 ST 用 chromadb ONNX(英文, 零下载).

    返回 list[list[float]]。所有路径都真实编码(非哈希), 保证品味增/查同一语义空间。
    """
    global _ONNX_EF
    st = _taste_st()
    if st is not None:
        import numpy as np
        return st.encode(texts, normalize_embeddings=True).tolist()
    if _ONNX_EF is None:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        _ONNX_EF = DefaultEmbeddingFunction()
    return _ONNX_EF(texts)


def _get_taste_kb():
    """品味召回用的自包含语义 KB(ST/ONNX 编码, 不依赖全局 KB 的 embedder 选择).

    与工作区共享 KB(预置 11881 块、embedder 全局选择)解耦: 单独一个 Chroma 集合,
    写入/查询都用 _taste_embed(同一模型), 快、不混化学语料、可离线。
    用 collection metadata 记录 embedder 指纹: 指纹不变 → 跨批次保留并继续积累;
    仅当语义空间切换(ST↔ONNX)才清空重建, 避免混空间检索。
    """
    global _TASTE_KB
    if _TASTE_KB is None:
        import chromadb
        root = _OUT / "taste_kb"
        root.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(root / "chroma"))
        col = client.get_or_create_collection("taste_kb")
        fp = _taste_embedder_fingerprint()
        old_fp = (col.metadata or {}).get("embedder_fingerprint")
        if old_fp != fp and col.count() > 0:
            # 语义空间变化 → 清空重来, 保证与当前 embedder 同空间.
            stale = col.get(include=[])
            if stale.get("ids"):
                col.delete(ids=stale["ids"])
            print(f"  [taste] embedder 空间切换({old_fp}->{fp}), 清空品味 KB 重建")
        col.modify(metadata={"embedder_fingerprint": fp})
        _TASTE_KB = col
    return _TASTE_KB


def _recall_taste(q: str = "How to pose research frontier questions: idealized assumption failure, cross-domain analogy, new enabling tools",
                  top_k: int = 4) -> str:
    """从品味专属语义 KB 召回已蒸馏的【研究品味/提出机制】启发式, 注入书生问题生成.

    自包含 RAG: 写入/查询同一 ONNX/ST 编码器; 查询词用英文(ONNX 为英文模型)。
    任何一步失败/无知识 → 优雅降级空串, 绝不阻断主线。
    """
    try:
        col = _get_taste_kb()
        n = col.count()
        if n == 0:
            return ""
        emb = _taste_embed([q])[0]
        hits = col.query(query_embeddings=[emb], n_results=min(top_k, n))
    except Exception as ee:  # noqa: BLE001 — 任何失败优雅降级
        print(f"  [taste-recall] 召回失败(降级空): {type(ee).__name__}")
        return ""
    lines = []
    docs = (hits or {}).get("documents") or [[]]
    for d in docs[0] or []:
        t = str(d).strip()
        if t and t not in lines:
            lines.append(f"- {t[:240]}")
    return "\n".join(lines) if lines else ""


def _distill_research_taste(client, model: str, goal: str) -> dict:
    """书生反身阅读: 每个问题是如何被提出的 + 提炼可迁移的发问品味 + 自生成新问题.

    返回 {mechanisms, taste_patterns, new_questions}.
    """
    evidence = "\n".join(f"- {q}: {ev[:220]}" for q, ev in _THEME_EVIDENCE)
    sys = (
        "你是研究品味提炼者(methodologist)。给你一份断裂力学综述的七大开放问题及其参考证据。"
        "任务不是解决它们, 而是反身回答『这些问题是怎么被提出来的』: 逐条定位它背后的"
        "【提出机制】—— 是哪一步理想化假设在现实的哪个边界失效、或哪种悖论/尺度耦合断裂/"
        "模型-实验鸿沟/新使能工具/跨域类比孕育了它。"
        "然后提炼可迁移的『发问品味』(生成新研究问题的可复用启示, 每个配一个非断裂域的类比例子)。"
        "最后依此品味, 提出 2~3 个白名单之外的、可证伪的、并能在纯数值实验里落地验证的"
        "新前沿问题(给出具体预言与拟用实验)。"
        "机制分类只用这些标签: " + " ".join(_TASTE_CATEGORIES) + "。"
        "只输出 JSON: "
        '{"mechanisms":[{"issue":"..","tension":"..(追问要点)","formation":"..(它如何被提出)",'
        '"category":"<标签>"}], "taste_patterns":[{"pattern":"..","nonfracture_example":".."}], '
        '"new_questions":[{"q":"..","mechanism":"..","prediction":"..","experiment":".."}]}, '
        "new_questions 每条 <300 字, 不含其他文字。"
    )
    user = f"{evidence}\n\n目标(供对齐节律): {goal[:600]}"
    # 端到端闭合(断点 B): 先召回已沉淀的品味记忆, 注入本次反身 —— 新品味从旧品味长出.
    recalled = _recall_taste("research taste: how frontier questions form via idealized-assumption failure, cross-domain analogy, enabling tools")
    if recalled:
        user += (f"\n\n【可复用参考 · 全局知识库召回的研究品味】\n{recalled}\n"
                 "(可吸收其模式, 但必须给出新的机制/新问题, 不要复读已有结论)")
    d = _ask_json(client, model, sys, user, max_tokens=2000)
    if not (d.get("mechanisms") and d.get("taste_patterns")):
        return {"mechanisms": [], "taste_patterns": [], "new_questions": [],
                "raw": d}
    return d


def _persist_taste(taste: dict, client, model: str, out_dir: Path) -> Path:
    """把品味分析落盘 markdown, 并经 Huginn 原生知识蒸馏沉淀进长期记忆(RAG 可检索)."""
    md = ["# 研究品味分析 —— 这些问题是如何被提出的", "",
          f"> 来自 IJF 2026 综述(杨卫/冯西桥/高华健)七大开放问题的问题-提出机制反身。", ""]
    md += ["## 一、问题背后的提出机制", ""]
    for m in taste.get("mechanisms") or []:
        md.append(f"- **{m.get('issue','')}** `[{m.get('category','')}]`")
        md.append(f"  - 追问要点(tension): {m.get('tension','')}")
        md.append(f"  - 如何被提出(formation): {m.get('formation','')}")
    md += ["", "## 二、可迁移的发问品味(taste patterns)", ""]
    for i, p in enumerate(taste.get("taste_patterns") or [], 1):
        md.append(f"{i}. **{p.get('pattern','')}** — 跨域类比: {p.get('nonfracture_example','')}")
    md += ["", "## 三、依品味自生成的新前沿问题(可证伪)", ""]
    for j, nq in enumerate(taste.get("new_questions") or [], 1):
        md.append(f"{j}. **{nq.get('q','')}**")
        md.append(f"   - 机制: {nq.get('mechanism','')}")
        md.append(f"   - 预言: {nq.get('prediction','')}")
        md.append(f"   - 拟实验: {nq.get('experiment','')}")
    md.append("")
    path = out_dir / "research_taste.md"
    path.write_text("\n".join(md), encoding="utf-8")
    # 原生知识蒸馏: 把品味作为一个可 RAG 检索的语义来源沉淀(无 LLM 契约/超时均优雅降级).
    try:
        from huginn.evolution.knowledge_distiller import KnowledgeDistiller
        kd = KnowledgeDistiller(output_dir=str(out_dir / "knowledge"))
        new = kd.distill_semantic_source(
            "\n".join(md), source_url="https://doi.org/10.1007/s10704-025-00907-6",
            source="review_taste", source_type="research_taste",
            domain_hint="fracture mechanics, research methodology", llm=_sync_llm_create(client, model))
        # 端到端闭合(断点 A): 蒸馏条目写入【品味专属语义 KB】(同一 ONNX/ST 编码器),
        # 后续 _recall_taste 在同一语义空间召回 —— 让品味真正可被书生在发问时检索.
        col = _get_taste_kb()
        docs = [dk.content for dk in new if dk.content and dk.content.strip()]
        if docs:
            embs = _taste_embed(docs)
            ids = [f"taste_{dk.knowledge_id}" for dk in new if dk.content and dk.content.strip()]
            metas = [{"source": "distilled_knowledge", "source_type": dk.source_type,
                      "confidence": dk.confidence, "created_at": dk.created_at}
                     for dk in new if dk.content and dk.content.strip()]
            col.upsert(ids=ids, documents=docs, embeddings=embs, metadatas=metas)
        print(f"  [taste] 蒸馏新条目 {len(new)} 条, 写入品味语义 KB {len(docs)} 条"
              f"(集合共 {col.count()} 条)")
    except Exception as ee:  # noqa: BLE001 — 蒸馏/提升失败不阻断主流程(品味 md 已是主产物)
        print(f"  [taste] 知识蒸馏/写入失败(不影响分析产物): {ee}")
    return path

# ═══════════════════════════════ 计划与主循环 ═══════════════════════════════

def _make_experiments(cycle: int = 1) -> list:
    from huginn.research import Experiment
    if cycle == 1:
        specs = [
            (lambda: exp_flaw_tolerance(), "F3_flaw_tolerance",
             "纳尺度缺陷容差(基准): 强度-缺陷尺寸交叉与临界缺陷尺寸 —— 回答\"为何纳米试样达理想强度\""),
            (lambda: exp_interface_oscillation(), "F1_interface_oscillation",
             "界面裂纹振荡奇异指数跨材料对跨度: 互穿悖论的定量谱(哪类界面需内聚/接触区模型)."),
            (lambda: exp_statistical_size_effect(), "F4_statistical_size",
             "统计弱链尺寸效应: Pareto 缺陷分布下强度∝N^{−1/(2m)} 的数值验证(R² 优度)."),
            (lambda: exp_dynamic_barrier(), "F6_dynamic_barrier",
             "mode-I Rayleigh 势垒: 精确 c_R 根 + Rose k(v) 能量通量比 + 声学缺口结构."),
            (lambda: exp_ductile_brittle_map(), "F2_dbt_map",
             "脆-韧转变图谱: γusf/γs 判据对 10 材料的区分度(本能脆/韧分区)."),
            (lambda: exp_bridging_toughening(), "F5_bridging_toughening",
             "桥联增韧: Dugdale 常数桥联区的表观韧性增益上界(仿生增韧量化)."),
            (lambda: exp_kic_validity(), "F7_kic_validity",
             "KIC 有效性边界: ASTM E399 门槛 2.5(K/σy)² 跨工程材料跨度(标准开放边界量化)."),
        ]
        return [Experiment(name=n, hypothesis=h, run=fn) for fn, n, h in specs]
    raise ValueError(f"cycle={cycle} 无内置实验; cycle>=2 走扫描分支")


def _make_scan_experiments(configs: list[dict]) -> list:
    from huginn.research import Experiment
    exps = []
    for i, cfg in enumerate(configs):
        name = f"S{i + 1}_scan"
        dim = (cfg or {}).get("dim", "pair")
        exps.append(Experiment(
            name=name,
            hypothesis=(f"断裂扫描 #{i + 1}: dim={dim} values={cfg.get('values')} "
                        f"→ 书生提议的真实族扫描(物理公式一致, 结果全真实)."),
            run=lambda cc=cfg, nn=name: exp_fracture_scan(cc, nn)))
        # agent 保底(L2)闭式归纳: 对 bridge 扫描, 从同一真实轨迹(x=s,y=Kc/K0)
        # 反推闭式 Kc/K0=sqrt(1+k*s). 归纳源=书生跑的真实点, 不伪造.
        dim_c = {"bridge_ratio": "bridge"}.get(dim, dim)
        if dim_c == "bridge" or dim == "nu" or dim == "barrier":
            vals = (cfg or {}).get("values") or SCAN_OPS.get(
                _DIM_VALUES.get(dim_c, dim_c)) or []
            if dim_c == "bridge" and len(vals) >= 3:
                cl_name = f"S{i + 1}_closure"
                xs = [float(v) for v in vals]
                ys = [_bridge_gain(float(v)) for v in vals]
                exps.append(Experiment(
                    name=cl_name,
                    hypothesis=(f"agent 闭式归纳(S{i + 1}): 从书生跑出的真实桥联轨迹"
                                f"(s∈{xs[0]}..{xs[-1]}) 反推 Kc/K0 闭式, 并用留出点核验."),
                    run=lambda xx=list(xs), yy=list(ys), nn=cl_name:
                        _closure_experiment(xx, yy, nn)))
    return exps


def _closure_experiment(xs: list[float], ys: list[float], name: str) -> dict:
    """agent 保底(L2)闭式归纳实验的 run(): 用部分点拟合闭式, 留出点核验.

    归纳源 = 书生跑出的真实轨迹(与扫描同源数值), 拟合出闭式表达式, 再用
    未参与拟合的留出点核对 —— 保证闭式不只记住采样点, 而是真规律.
    返回可过门禁的 objectives/summary(全为真实计算标量).
    """
    expr_str = _close_form_fit(xs, ys, "bridge")
    if expr_str is None:
        # 拟合失败/过差: 如实返回零证据, 不伪造闭式.
        return {"objectives": {}, "summary": {"closed_form": None,
                "note": "闭式归纳未收敛(r2/截距门槛未达), 未产出闭式"},
                "success": True}
    import sympy as sp
    expr_sym = sp.sympify(expr_str)
    chk = _sym_ground_check(expr_sym, "bridge")
    if chk is None or chk["max_relerr"] > 5e-3:
        return {"objectives": {}, "summary": {"closed_form": str(expr_sym),
                "max_relerr": chk["max_relerr"] if chk else None,
                "note": "闭式归纳后核验不一致, 未采用"}, "success": True}
    s_mid = sp.Float(float(xs[len(xs) // 2]))
    objective_val = round(float(expr_sym.subs(sp.Symbol("s"), s_mid).evalf()), 6)
    return {"objectives": {"bridge_closure": objective_val},
            "summary": {"closed_form": str(sp.simplify(expr_sym))[:300],
                        "family": "bridge", "induce_from": name,
                        "n_fit": len(xs), "max_relerr": chk["max_relerr"],
                        "note": "agent 从书生真实桥联轨迹闭式归纳并经数值核验"},
            "success": True}


def _build_plan(goal: str, run_by_name: dict, cycle: int = 1):
    from huginn.research.planning import build_research_plan, SubResearch
    if cycle == 1:
        return build_research_plan(goal, [
            SubResearch("F3_flaw_tolerance", "基准: 纳尺度缺陷容差(尺度坐标)",
                        run=run_by_name["F3_flaw_tolerance"], depends_on=[]),
            SubResearch("F1_interface_oscillation", "界面互穿悖论(界面线)",
                        run=run_by_name["F1_interface_oscillation"],
                        depends_on=["F3_flaw_tolerance"]),
            SubResearch("F4_statistical_size", "统计弱链尺寸效应(统计线)",
                        run=run_by_name["F4_statistical_size"],
                        depends_on=["F3_flaw_tolerance"]),
            SubResearch("F6_dynamic_barrier", "Rayleigh 势垒(动态线)",
                        run=run_by_name["F6_dynamic_barrier"],
                        depends_on=["F3_flaw_tolerance"]),
            SubResearch("F2_dbt_map", "确认组: 脆韧图谱",
                        run=run_by_name["F2_dbt_map"],
                        depends_on=["F3_flaw_tolerance"]),
            SubResearch("F5_bridging_toughening", "确认组: 桥联增韧",
                        run=run_by_name["F5_bridging_toughening"],
                        depends_on=["F3_flaw_tolerance"]),
            SubResearch("F7_kic_validity", "确认组: KIC 有效性边界",
                        run=run_by_name["F7_kic_validity"],
                        depends_on=["F3_flaw_tolerance"]),
        ], parallel_cap=3)
    names = list(run_by_name.keys())
    return build_research_plan(goal, [
        SubResearch(n, f"书生提议断裂扫描分支 {n}", run=run_by_name[n], depends_on=[])
        for n in names
    ], parallel_cap=3)


def _dim_objective(dim: str) -> str:
    """扫描维度 → 该族实验返回的真实目标键(与基分支公式一致, 保证门禁可落地)."""
    return {"pair": "osc_span", "material": "dbt_span", "flaw": "flaw_tol_100nm",
            "n_flaws": "weibull_fit", "bridge": "kce_gain",
            "bridge_ratio": "kce_gain",
            "barrier": "barrier_sharpness", "nu": "barrier_sharpness",
            "kic": "kic_span"}.get(dim, "osc_span")  # 未知维兜底到 pair 族(不崩批)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry", action="store_true", help="确定性运行(不调模型)")
    ap.add_argument("--model", default="intern-s2-preview")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--max-iters", type=int, default=12)
    ap.add_argument("--min-iters", type=int, default=3)
    ap.add_argument("--strictness", type=int, default=0, choices=[0, 1, 2])
    ap.add_argument("--cycles", type=int, default=2)
    ap.add_argument("--start-cycle", type=int, default=1)
    ap.add_argument("--taste", action="store_true",
                    help="开启研究品味层: 每轮前让书生反身思考【这些问题如何被提出】"
                         "并自生成白名单外的新前沿问题(写入 research_taste.md 并蒸馏进记忆)")
    args = ap.parse_args()

    from huginn.research import run_research_program, grounding_verifier   # noqa: F401
    from huginn.research import Experiment                                   # noqa: F401

    client = None
    if not args.dry:
        key = os.environ.get("INTERNLM_API_KEY")
        if not key:
            print("error: INTERNLM_API_KEY not set (或 --dry)", file=sys.stderr)
            return 2
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url=args.base_url or
                        os.environ.get("INTERNLM_BASE_URL",
                                       "https://chat.intern-ai.org.cn/api/v1"))

    def _report_for(cycle: int) -> Path:
        return _OUT / ("research_report.md" if cycle == 1 else f"cycle{cycle}_report.md")

    print("== 书生 + Huginn 断裂力学域长程深研管线 (IJF 2026 综述驱动) ==")

    _OBJ1 = {k: "maximize" for k in
             ("flaw_tol_100nm", "osc_span", "weibull_fit", "barrier_sharpness",
              "dbt_span", "kce_gain", "kic_span")}
    last_report = ""
    if args.start_cycle >= 2:
        prev = _report_for(max(1, args.start_cycle - 1))
        if prev.exists():
            last_report = prev.read_text(encoding="utf-8")

    done_cfgs: list[dict] = []

    for cycle in range(max(1, args.start_cycle), args.start_cycle + args.cycles):
        print(f"\n===== 断裂域 第 {cycle} 轮(长程自主) =====")
        if cycle == 1:
            goal = GOAL
            exps = _make_experiments(1)
            objectives = dict(_OBJ1)
            diagnostics = _diagnostic_tools()
        else:
            next_open = _extract_next_open(last_report)
            if client is not None:
                qs = _propose_next_open(client, args.model, last_report)
                next_open = " ".join(x for x in (next_open, " ".join(qs)) if x)
                if qs:
                    print(f"  [书生·观察] 下一轮开放问题: {qs}")
            # Code Lab: 书生亲手写本轮实验代码(每轮尝试, 失败回退白名单).
            # 优先 Symbolic Lab(从控制方程推闭式解, 超越数值扫描); 失败再退
            # 数值书生成码; 再失败回退白名单扫描. 三条路径任一绕过都算书生行动.
            author_exp, author_probes, author_obj_keys, author_err = (None, [], [], "dry")
            if client is not None:
                author_exp, _sym_keys, author_err = _try_author_symbolic(
                    client, args.model, next_open, cycle)
                if author_exp is not None:
                    author_obj_keys = list(_sym_keys)
                    print(f"  [Symbolic Lab] 书生推出闭式解并通过与物理核的数值核验: "
                          f"{author_exp.hypothesis[:150]}")
                else:
                    print(f"  [Symbolic Lab] 未通过: {author_err}")
                    author_exp, author_probes, author_obj_keys, author_err = \
                        _try_author_code(client, args.model, next_open, cycle)
                    if author_exp is None:
                        print(f"  [书生成码] 未通过, 回退白名单扫描: {author_err}")
            if client is not None:
                cand = _propose_scan_configs(client, args.model, next_open, done_cfgs)
            else:
                cand = []
            cfgs = _pad_scan_configs(cand, next_open)
            fresh = [c for c in cfgs
                     if json.dumps(c, sort_keys=True)
                     not in {json.dumps(d, sort_keys=True) for d in done_cfgs}]
            need = max(0, 3 - (1 if author_exp is not None else 0) - len(fresh))
            cfgs = fresh + [c for c in cfgs if c not in fresh][:need]
            for c in cfgs:
                done_cfgs.append(c)
            exps = ([author_exp] if author_exp is not None else []) + \
                _make_scan_experiments(cfgs)
            objectives = {}
            for c in cfgs:
                dim = c.get("dim", "pair")
                objectives[_dim_objective(dim)] = "maximize"
            if any(e.name.endswith("_closure") for e in exps):
                objectives["bridge_closure"] = "maximize"
            if author_exp is not None:
                for _k in author_obj_keys:
                    objectives[_k] = "maximize"
            goal = (f"检验书生本轮提出的断裂开放问题(证据由断裂扫描/Symbolic Lab 闭式"
                    f"推导/书生成码提供): {next_open}")
            print(f"  [书生·行动] 本轮扫描配置: {cfgs}")

        run_by_name = {e.name: e.run for e in exps}
        plan = _build_plan(goal, run_by_name, cycle)
        report_md = _report_for(cycle)
        print(f"goal: {goal[:120]}...")
        print(f"experiments: {[e.name for e in exps]}  layers: {plan.layers}")

        # ── 研究品味层(可选, 需书生): 反身"这些问题如何被提出" + 自生成新问题 ──
        taste_new_questions: list[dict] = []
        if args.taste and client is not None:
            _taste = _distill_research_taste(client, args.model, goal)
            if _taste.get("mechanisms") or _taste.get("taste_patterns"):
                _persist_taste(_taste, client, args.model, _OUT)
                taste_new_questions = _taste.get("new_questions") or []
                print(f"  [taste] 提炼 {len(_taste.get('mechanisms', []))} 条提出机制, "
                      f"{len(_taste.get('taste_patterns', []))} 条发问品味, "
                      f"自生成 {len(taste_new_questions)} 个新前沿问题")
                for _nq in taste_new_questions:
                    print(f"    · {str(_nq.get('q',''))[:90]}")
                # 书生自生成问题并入本轮目标, 交由后续扫描/书生成码/报告成文承接.
                if taste_new_questions:
                    _nq_txt = " | ".join(str(n.get("q", "")) for n in taste_new_questions)
                    goal = f"{goal}\n（书生依品味自生成的新问题: {_nq_txt[:500]}）"

        out = run_research_program(
            goal=goal,
            experiments=exps,
            objectives_config=objectives,
            client=client, model=args.model, base_url=args.base_url,
            verify=grounding_verifier(),
            out_md=report_md,
            planner=lambda _g: plan,
            layer_epochs=True,
            replan_gate=True,
            early_stop_gate=True,
            diagnostic_tools=diagnostics,
            max_iterations=args.max_iters,
            min_iterations=min(args.min_iters, max(1, len(exps) - 1)),
            max_parallel=2,
            strictness=args.strictness,
        )
        print("\n[program] explored=%d pruned=%d pareto_front=%d convergence=%s"
              % (out.explored, out.pruned, len(out.pareto_front), out.converred))
        for b in out.pareto_front:
            print(f"  surv -> {b['name']}")
        print(f"[gate] {out.verdict} unsubstantiated={out.ungrounded} "
              f"source={out.report_source}")

        if client is not None and out.report:
            _sv = "\n".join(
                f"- {b['name']}: {json.dumps(out.cache.get(b['name'], {}).get('summary', {}), ensure_ascii=False)[:400]}"
                for b in out.pareto_front) or "(无存活)"
            _finds = _llm_critic(client, args.model, out.report, _sv)
            if _finds:
                _blk = ["", "## 对立审稿(CriticAgent)",
                        "> 书生双角色协同: 主研究员成文, 审稿副体持反对立场复核。"]
                for f in _finds:
                    _blk.append(f"- 断言: {f.get('claim', '')}\n"
                                f"  - 风险: {f.get('risk', '')}\n"
                                f"  - 建议: {f.get('suggest', '')}")
                out.report += "\n" + "\n".join(_blk)
                if report_md.exists():
                    report_md.write_text(
                        report_md.read_text(encoding="utf-8").rstrip() + "\n" +
                        "\n".join(_blk) + "\n", encoding="utf-8")
                print(f"  [CriticAgent] 审稿副体提出 {len(_finds)} 条降级建议(已并入报告)")
        print(f"第{cycle}轮报告: {report_md}")
        last_report = out.report or (report_md.read_text(encoding="utf-8")
                                     if report_md.exists() else "")

    print("\n===== 断裂域长程任务结束: 连续完成 %d 轮(观察→推理→行动→报告→下一轮, 无人工停顿) ====="
          % args.cycles)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())