"""书生 Intern-S2 + Huginn 完整深研管线(非旁路, 全走 run_research_program).

上一版(`shusheng_agent_nn_rigidity`)是**旁路的裸工具循环**: 只借了
`grounding_verifier`, 选题->实验->成文全部由模型自己在一个循环里对各独立工具做
tool-call, 没有进入 Huginn 的**科研工作流**本身。本文件纠正这一点:
书生作为 LLM 引擎**驱动 `huginn.research.run_research_program`**, 完整走
Huginn 既定工作流:

  1. 真实证据实验 (X1 eps判据 / X2 优化器财政 / X3 约束维数) 注册为 `Experiment`,
     每个分支返回独立目标(objective) + 完整 summary —— 数值全部来自真实计算;
  2. `planner`(需求拆解)把 goal 拆成带依赖的 DAG → 分层:
       层0: X1 (判据基准);  层1: X2, X3 (并行确认, 依赖 X1)
     → 触发 P-A 分层流式结算(layer_epochs) + P-B 层间重规划门(replan_gate)
          + P-C 证据驱动提前终止门(early_stop_gate);
  3. 策略链: Pareto 剪枝 (淘汰被支配假说; 三条独立证据维度互不支配 → 全存活,
        保住多证据方向). 本 demo 刻意不开变异(mutation): 三条实验是独立子结论,
        不是连续参数搜索空间, 变异只会制造与真实轨迹对不上的假说噪声;
        需参数寻优的研究者在上层 open 即可恢复 evolution 闭环;
  4. 书生拿到存活假说真实证据, **自主调用域诊断工具**(diagnostic_tools 里的
     probe_* )采集补充证据, 批判综合成报告;
  5. 声明门禁(claim_grounding) 核对报告每个数值都落在工具轨迹里, 未落地则要求重写;
  6. 真机交不出可落地报告 → 确定性兜底组装(数值全来自真实执行, 必过门禁).

设计约束(诚实边界):
  - 所有 run/工具返回 objectives 与 summary 都由真实计算产生, 无任何编造;
  - objectives 各分支独立(各自支撑不同子结论), 互不支配 → 三个证据分支全部存活,
    保住"多证据方向"而非被单一维度平凡淘汰;
  - `--dry` 走 client=None 确定性综合路径, 无 API key 也能验证整条管线能跑通。

用法:
  INTERNLM_API_KEY=... python examples/shusheng_huginn_workflow.py          # 书生全流程
  python examples/shusheng_huginn_workflow.py --dry                         # 确定性验证管线
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

_AGENT = Path(__file__).resolve().parents[1] / "agent"
if str(_AGENT) not in sys.path:
    sys.path.insert(0, str(_AGENT))

_OUT = Path(__file__).resolve().parent / "out" / "shusheng_huginn_workflow"
_OUT.mkdir(parents=True, exist_ok=True)

GOAL = (
    "深入 X7 开放问题: 约束钳制解族维数的机制与边界。上一轮 X7 报告'多项式系统 u''=-(1-t^2) "
    "下 σH_k2 不归零(普适性减弱)'——本轮先证伪/确证该异常是否是实验定义伪影; 再研究真正开放的机制: "
    "k=1 中间态(σH 只降 ~54%)下 1 个点值约束钳制哪个核方向 {常数, 线性}, 以及它随约束位置取值的 "
    "依赖(约束在中心 t_i≈0 时几乎测不到线性核方向)。要求: 所有数值真实可复现, 区分'实验伪影'与'真实机制'。"
)

# 每个实验独立目标(各自支撑不同子结论; 独立维度互不支配 → 全部存活).
# 本轮新增: 普适性真伪(修正 ref 后)/k=1 中间态对约束位置的依赖.
ACCURACY_OBJ = ["eps_discrimination", "n_stability", "budget_convergence",
                "dim_effect", "univ_effect", "k1_locality"]
_OBJECTIVES = {k: "maximize" for k in ACCURACY_OBJ}

# ── 第二轮(X9/X10/X11): 核方向分解 —— 上一轮报告"下一步"之首 ────────────
GOAL_CYCLE2 = (
    "深挖 k=1 中间态的核方向钳制机制(上轮报告'下一步'第2/3/4条): "
    "(1) 显式把残余零空间方向分解到常数核 P0=1 与线性核 P1=t, 确认约束位置 t_i 如何决定 "
    "钳制哪个核方向 —— 理论预言: 约束 u(t_i)=ref 在解族 u=cos(t)+C1+C2·t 上等价于 C1=-C2·t_i, "
    "残余方向 = (t-t_i), 即 lin_frac = 1/(1+t_i^2)(中心≈1 纯线性核; 边界≈0.52 常数+线性混合); "
    "(2) 建立位置→方向占比的定量曲线并检验该定律; "
    "(3) 跨系统(cos/sin2t/多项式)验证该方向定律普适。同时用分解数据复核上一轮 X8 解读: "
    "'中心约束几乎测不到线性核方向'是真实机制还是归一化伪影。要求所有数值真实可复现。"
)
KERNEL_OBJ = ["k1_direction_purity", "locality_law_fit", "direction_universality"]
_OBJECTIVES_CYCLE2 = {k: "maximize" for k in KERNEL_OBJ}

# 长程驱动: 通用方向扫描能力白名单(书生在第 3+ 轮据此提议新实验配置).
# 每个配置都映射到真实执行(任意阶 ODE 的约束钳制 → 核方向分解), 无任何编造.
# order=2 → 核 span{1,t}(2维); order=3 → 核 span{1,t,t²}(3维, 高次核).
# ctype=point → 值约束 u(t_i); ctype=derivative → 导数约束 u'(t_i).
SCAN_OPS = {
    "system": [0, 1, 2],                    # 0=cos, 1=sin2t, 2=多项式
    "k": [1, 2],                            # 约束数
    "positions": [0.0, 0.33, 0.66, 0.97, -0.33, -0.66, -0.97],
    "basis": [8, 12, 16],
    "seeds": [4, 8, 12],
    "order": [2, 3],                        # ODE 阶数(2=低次核, 3=高次核)
    "ctype": ["point", "derivative"],       # 约束类型(值/导数)
}
_SCAN_DEFAULTS = {"system": 0, "k": 1, "positions": [0.0, 0.97],
                  "basis": 12, "seeds": 8, "order": 2, "ctype": "point"}

# 收敛聚合头头数预算等通用治理参数走 run_research_program 默认, 本 demo 不再重复.

# ── 真实证据实验 (run 返回 {objectives, summary, success}) ────────────────
_RNG = np.random.default_rng(0)

# 多系统表 (label, rhs, ref, refp) —— X7/X11+ 共用, 一处定义不漂移.
# refp = ref 的解析导数(导数约束 u'(t_i)=ref'(t_i) 用).
_SYSTEMS = [
    ("u''=-cos(t), ref=cos", lambda t: -np.cos(t), lambda t: np.cos(t),
     lambda t: -np.sin(t)),
    ("u''=-sin(2t), ref=sin(2t)/4", lambda t: -np.sin(2 * t),
     lambda t: np.sin(2 * t) / 4.0, lambda t: np.cos(2 * t) / 2.0),
    ("u''=-(1-t^2), ref=t^4/12-t^2/2", lambda t: -(1 - t ** 2),
     lambda t: t ** 4 / 12.0 - t ** 2 / 2.0, lambda t: t ** 3 / 3.0 - t),
]

# 勒让德算子缓存: (basis, order) -> (mesh, P, D_order), 供各实验复用(同一真实算子).
_OP_CACHE: dict = {}


def _build_system(basis: int, N: int = 256):
    """order=2 算子缓存(兼容旧调用; 高阶走 _build_system_g)."""
    return _build_system_g(basis, order=2, N=N)


def _build_system_g(basis: int, order: int = 2, N: int = 256):
    """勒让德基础算子: (mesh, P, D_order) —— P 为基函数取值矩阵, D_order 为阶导数.

    order=2 → 二阶微分算子(D2), 核 span{1,t} (2 维);
    order=3 → 三阶微分算子(D3), 核 span{1,t,t²} 的 Legendre 版 (3 维).
    """
    from numpy.polynomial import legendre as Lg
    import numpy.polynomial.polynomial as PP
    key = (basis, order)
    if key in _OP_CACHE:
        return _OP_CACHE[key]
    mesh = np.linspace(-1, 1, N)
    P = Lg.legval(mesh, np.eye(basis)).T
    D = np.zeros((N, basis))
    for j in range(basis):
        d = PP.polyder(Lg.leg2poly(np.eye(j + 1)[:, j]), order)
        acc = np.zeros_like(mesh)
        for kk, cc in enumerate(d):
            acc += cc * mesh ** kk
        D[:, j] = acc
    _OP_CACHE[key] = (mesh, P, D)
    return mesh, P, D


def _build_p1(basis: int, N: int = 256) -> np.ndarray:
    """基函数一阶导矩阵 P1: P1[i, j] = d(P_j)/dt at mesh[i] (导数约束行用)."""
    from numpy.polynomial import legendre as Lg
    import numpy.polynomial.polynomial as PP
    mesh = np.linspace(-1, 1, N)
    P1 = np.zeros((N, basis))
    for j in range(basis):
        d1 = PP.polyder(Lg.leg2poly(np.eye(j + 1)[:, j]), 1)
        acc = np.zeros_like(mesh)
        for kk, cc in enumerate(d1):
            acc += cc * mesh ** kk
        P1[:, j] = acc
    return P1


def _poly_cstar(target: str, eps: float, N: int, pmax: int = 48) -> float:
    """固定 N 下, 达到留出精度 eps 所需最小偶多项式项数 (C* 容量单位). 真实最小二乘."""
    f = (lambda t: np.cos(t)) if target == "rigid" else (lambda t: np.abs(t))
    x = _RNG.uniform(-1.5, 1.5, N); y = f(x)
    xv = np.linspace(-1.5, 1.5, 1501); yv = f(xv)
    for k in range(1, (pmax // 2) + 1):
        A = np.stack([x ** (2 * j) for j in range(k)], axis=1)
        coef, *_ = np.linalg.lstsq(A.T @ A + 1e-10 * np.eye(k), A.T @ y, rcond=None)
        B = np.stack([xv ** (2 * j) for j in range(k)], axis=1)
        if float(np.mean((B @ coef - yv) ** 2)) <= eps:
            return float(k)
    return float(pmax // 2 + 1)


def exp_eps_criterion(N: int = 4096) -> dict:
    """X1 · eps判据: 固定 N 扫 eps, 刚性 cos vs 胖 |x| 的 C* 形态差异."""
    eps_list = [1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-10]
    c_rig = [_poly_cstar("rigid", e, N) for e in eps_list]
    c_fat = [_poly_cstar("fat", e, N) for e in eps_list]
    # 判据区分度: 在最小 eps 处 胖/刚 容量比 —— eps判据能否拉开两类系统(真实标量).
    disc = float(max(c_fat) / (max(c_rig) or 1.0))
    return {
        "objectives": {"eps_discrimination": disc},
        "summary": {"N": N, "eps": eps_list, "cstar_rigid": c_rig, "cstar_fat": c_fat,
                    "eps_discrimination": round(disc, 3)},
        "success": True,
    }


def _exp_optimizer_finance(w: int = 32, steps: int = 1600) -> dict:
    """X2 · 优化器财政: 真实 PINN 在训练预算下的留出误差 vho(刚性唯一解标定)."""
    import torch
    import torch.nn as nn
    torch.manual_seed(1)
    net = nn.Sequential(nn.Linear(1, w), nn.Tanh(), nn.Linear(w, w), nn.Tanh(),
                        nn.Linear(w, 1))
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    xc = torch.linspace(-1, 1, 60).reshape(-1, 1).requires_grad_(True)
    xp = torch.tensor([-1.0, 1.0]).reshape(-1, 1)
    yp = torch.cos(xp) + 0.22985 * xp + 0.7701
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    vtr_last = None
    for _ in range(steps):
        opt.zero_grad()
        u = net(xc).squeeze(-1)
        du = torch.autograd.grad(u, xc, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        ddu = torch.autograd.grad(du, xc, grad_outputs=torch.ones_like(du), create_graph=True)[0]
        res = ddu + torch.cos(xc.squeeze(-1))
        val = net(xp).squeeze(-1) - yp.squeeze(-1)
        L = res.pow(2).mean() + val.pow(2).mean()
        L.backward(); opt.step(); sch.step()
        vtr_last = float(L.item())
    xv = torch.linspace(-1, 1, 200).reshape(-1, 1)
    with torch.no_grad():
        uV = net(xv).squeeze(-1)
    uStar = torch.cos(xv).squeeze(-1) + 0.22985 * xv.squeeze(-1) + 0.7701
    vho = float(((uV - uStar).detach() ** 2).mean().sqrt() / (uStar ** 2).mean().sqrt())
    # 收敛证据: -log10(vho) 越大 = 越收敛(真实标量; vho 越小越好 → 取负对数最大化).
    conv = float(-math.log10(max(vho, 1e-12)))
    return {
        "objectives": {"budget_convergence": conv},
        "summary": {"w": w, "steps": steps, "vtr_last": round(vtr_last, 6),
                    "vho": round(vho, 6), "budget_convergence": round(conv, 3)},
        "success": True,
    }


def _solve_under(k: int, seed: int, basis: int,
                 rhs=lambda t: -np.cos(t), ref=lambda t: np.cos(t),
                 fixed_ti: float | None = None) -> np.ndarray:
    """欠定 u''=rhs(t) + k 个点值约束 u(t_i)=ref(t_i) -> 解族成员 (真实, 冻结留出采样).

    不做 min-norm 唯一解(lstsq 恒唯一 → 重解零散布, 测不出"解族维数"); 而是
    在约束线性系统 A c ≈ b 的**零空间**里随机取向, 叠加到特解上 —— 采样得到的
    是同一个真实解族(常数+线性 2 维核), k 个独立点值约束逐条钳制核自由度:
      k=0 -> 核维 2, 散布最大; k=2 -> 核被完全钳制, 散布 ~0. → 解族维数真测量.
    多系统普适性: 传不同 (rhs, ref) 即换一个真实 ODE 系统, 核恒为 2 维线性流形.
    fixed_ti(仅 k=1): 把唯一约束位置钉在 t≈fixed_ti, 检验位置如何决定钳制哪个核方向.
    """
    from numpy.polynomial import legendre as Lg
    mesh, P, D2 = _build_system(basis)
    N = mesh.shape[0]
    fvec = rhs(mesh)
    rng = np.random.default_rng(seed)
    A = D2.copy(); b = fvec.copy()
    if k > 0:
        if fixed_ti is not None and k == 1:
            idx = np.array([int(np.argmin(np.abs(mesh - fixed_ti)))])
        else:
            idx = rng.choice(N, size=k, replace=False)
        A = np.vstack([D2, 3.0 * P[idx]])
        b = np.concatenate([fvec, 3.0 * ref(mesh[idx])])
    # 特解(解族的一个成员) + 零空间采样(recover the真 family spread)
    c_particular, *_ = np.linalg.lstsq(A, b, rcond=None)
    _, s, vh = np.linalg.svd(A, full_matrices=True)
    rank = int(np.sum(s > 1e-8))
    null_basis = vh[rank:]                       # (null_dim, basis), 零空间方向
    if null_basis.shape[0] > 0:
        # 随机核组合, 归一化到与原特解同数量级 —— 采样的是各处留出上的真实解族展开
        coeff = rng.normal(0.0, 0.5, size=null_basis.shape[0])
        c_family = c_particular + coeff @ null_basis
    else:
        c_family = c_particular                   # 核被完全钳制 -> 唯一解
    xv = np.linspace(-1, 1, 321)
    return Lg.legval(xv, np.eye(basis)).T @ c_family


def exp_constraint_dimension(seeds: int = 8) -> dict:
    """X3 · 约束维数: k=0 相对 k=2 的解族展开 sigmaH 之差(真实解族维数探针)."""
    ev0 = np.stack([_solve_under(0, s, 12) for s in range(seeds)])
    ev2 = np.stack([_solve_under(2, s, 12) for s in range(seeds)])
    sig0 = float(ev0.std(axis=0).mean() / (np.abs(ev0).mean() + 1e-9))
    sig2 = float(ev2.std(axis=0).mean() / (np.abs(ev2).mean() + 1e-9))
    effect = float(sig0 - sig2)          # 约束把解族维数收掉的程度(真实标量, maximize)
    return {
        "objectives": {"dim_effect": effect},
        "summary": {"seeds": seeds, "sigmaH_k0": round(sig0, 4),
                    "sigmaH_k2": round(sig2, 4), "dim_effect": round(effect, 4)},
        "success": True,
    }


def exp_n_scale(steps_eps: float = 1e-4, Ns=(512, 1024, 2048, 4096, 8192)) -> dict:
    """X4 · N 平台: 固定 eps 扫 N, 刚性/胖的 C* 是否随 N 平台化(判据不依赖 N 的对照)."""
    rows = []
    span_rig, span_fat = [], []
    for N in Ns:
        c_rig = _poly_cstar("rigid", steps_eps, N)
        c_fat = _poly_cstar("fat", steps_eps, N)
        rows.append({"N": N, "Cstar_rigid": c_rig, "Cstar_fat": c_fat})
        span_rig.append(c_rig); span_fat.append(c_fat)
    # N 稳定性: C* 在 N 上的取值区间半径越小 → 对 N 越不敏感(平台化). 
    # 分辨率: N 跨度 8 倍, 若 C* 只 ±0 (完美平台) → 判据纯粹依赖 eps.
    rig_span = max(span_rig) - min(span_rig)
    fat_span = max(span_fat) - min(span_fat)
    n_stability = float(1.0 / (1.0 + rig_span + fat_span))   # 越大 = 对 N 越不敏感
    return {
        "objectives": {"n_stability": n_stability},
        "summary": {"eps": steps_eps, "Ns": list(Ns), "rows": rows,
                    "rig_span": rig_span, "fat_span": fat_span,
                    "n_stability": round(n_stability, 4)},
        "success": True,
    }


def exp_constraint_curve(seeds: int = 8, ks=(0, 1, 2, 3)) -> dict:
    """X5 · 约束全曲线: k=0..3 的 sigmaH 序列, 验证「约束数→解族维数」单调钳制曲线."""
    curve = []
    for k in ks:
        vals = np.stack([_solve_under(k, s, 12) for s in range(seeds)])
        sig = float(vals.std(axis=0).mean() / (np.abs(vals).mean() + 1e-9))
        curve.append({"k": k, "sigmaH": round(sig, 4)})
    # 钳制强度: k=0 到 k=3 的总下降量 (maximize) —— 约束逐条钳掉核自由度.
    strength = float(curve[0]["sigmaH"] - curve[-1]["sigmaH"])
    return {
        "objectives": {"dim_effect": strength},
        "summary": {"seeds": seeds, "curve": curve, "dim_effect": round(strength, 4)},
        "success": True,
    }


def exp_budget_sweep(w: int = 32, budgets=(400, 800, 1600, 3200, 6400)) -> dict:
    """X6 · 预算扫描: 训练预算递增下的 vho 收敛曲线(量化优化器财政假饱和窗口)."""
    rows = []
    for steps in budgets:
        r = _exp_optimizer_finance(w=w, steps=steps)
        rows.append({"steps": steps, "vho": r["summary"]["vho"],
                     "budget_convergence": r["summary"]["budget_convergence"]})
    # 假饱和窗口: (vho@min_budget - vho@max_budget) 相对改善 —— 预算不足会假饱和(污染判据).
    v0, v1 = rows[0]["vho"], rows[-1]["vho"]
    window = float(max(0.0, v0 - v1))
    return {
        "objectives": {"budget_convergence": rows[-1]["budget_convergence"]},
        "summary": {"w": w, "budgets": list(budgets), "rows": rows,
                    "vho_min_budget": round(v0, 6), "vho_max_budget": round(v1, 6),
                    "finance_window": round(window, 6)},
        "success": True,
    }


def exp_multi_system(seeds: int = 8) -> dict:
    """X7 · 多系统普适: 不同 ODE 系统下约束钳制解族维数的效应是否一致."""
    systems = [
        ("u''=-cos(t), ref=cos", lambda t: -np.cos(t), lambda t: np.cos(t)),
        ("u''=-sin(2t), ref=sin(2t)/4", lambda t: -np.sin(2 * t), lambda t: np.sin(2 * t) / 4.0),
        ("u''=-(1-t^2), ref=t^4/12-t^2/2", lambda t: -(1 - t ** 2), lambda t: t ** 4 / 12.0 - t ** 2 / 2.0),
    ]
    per = []
    for label, rhs, ref in systems:
        ev0 = np.stack([_solve_under(0, s, 12, rhs=rhs, ref=ref) for s in range(seeds)])
        ev2 = np.stack([_solve_under(2, s, 12, rhs=rhs, ref=ref) for s in range(seeds)])
        sig0 = float(ev0.std(axis=0).mean() / (np.abs(ev0).mean() + 1e-9))
        sig2 = float(ev2.std(axis=0).mean() / (np.abs(ev2).mean() + 1e-9))
        per.append({"system": label, "sigmaH_k0": round(sig0, 4),
                    "sigmaH_k2": round(sig2, 4), "effect": round(sig0 - sig2, 4)})
    effects = [p["effect"] for p in per]
    univ = float(min(effects))          # 最弱系统也有正效应 → 普适(保守最大化)
    return {
        "objectives": {"univ_effect": univ},
        "summary": {"seeds": seeds, "systems": per, "min_effect": univ,
                    "mean_effect": round(float(np.mean(effects)), 4)},
        "success": True,
    }


def exp_k1_locality(seeds: int = 8, basis: int = 12) -> dict:
    """X8 · k=1 中间态: 约束位置 t_i 决定 1 个点值约束钳制核的哪个方向.

    欠定 u''=-cos 的核 = span{1, t}(2 维). k=1 留 1 维残余自由. 位置的作用(开放问题, 待实测):
      中心 t_i≈0 时, 约束 u(0)=ref ≈ C1·1, 对线性核 C2·t 几乎无感 → 钳常数、留纯线性核;
      边界 t_i≈±1 时, u(t_i) 含满 C2·t_i 之幅 → 同时钳常数+线性(混合方向).
    但 σH 是**归一化相对量**(std/|mean|), 差异既来自绝对散布也来自分母 |mean| ——
    须同时报告绝对 std, 才能区分"钳制力差异"与"归一化/核形状效应".
    """
    def sigma_at(t_center: float) -> tuple[float, float, float]:
        evs = []
        for s in range(seeds):
            rng = np.random.default_rng(s)
            t_i = float(np.clip(t_center + rng.uniform(-0.02, 0.02), -1, 1))
            evs.append(_solve_under(1, s, basis, rhs=lambda t: -np.cos(t),
                                    ref=lambda t: np.cos(t), fixed_ti=t_i))
        evs = np.stack(evs)
        abs_std = float(evs.std(axis=0).mean())
        abs_mean = float(np.abs(evs).mean())
        return abs_std, abs_mean, float(abs_std / (abs_mean + 1e-9))

    c_std, c_mean, c_ratio = sigma_at(0.0)
    b_std, b_mean, b_ratio = sigma_at(0.97)
    loc_norm = float(c_ratio - b_ratio)
    loc_abs = float(c_std - b_std)        # 绝对散布差(排除归一化影响)
    return {
        "objectives": {"k1_locality": loc_norm},
        "summary": {"seeds": seeds,
                    "center": {"abs_std": round(c_std, 4), "abs_mean": round(c_mean, 4),
                               "sigmaH": round(c_ratio, 4)},
                    "boundary": {"abs_std": round(b_std, 4), "abs_mean": round(b_mean, 4),
                                 "sigmaH": round(b_ratio, 4)},
                    "k1_locality_norm": round(loc_norm, 4),
                    "k1_locality_abs": round(loc_abs, 4)},
        "success": True,
    }


def _solve_under_decomp(k: int, seed: int, basis: int,
                        rhs=lambda t: -np.cos(t), ref=lambda t: np.cos(t),
                        fixed_ti: float | None = None):
    """同 _solve_under, 但额外返回零空间方向 v(勒让德系数空间).

    v 是欠定系统 [D2; 3P(t_i)] 的零空间方向(确定性, 由约束位置决定):
    对 u''=rhs 齐次核 = span{1,t}, 加 1 个点值约束后 v 落在 (P0, P1) 子空间,
    且等价于 (t - t_i) 方向(理论: u=particular+C1+C2·t, 约束 ⇒ C1=-C2·t_i).
    返回 (uvals, v) 或 (uvals, None)(核已被完全钳制时无零空间).
    """
    from numpy.polynomial import legendre as Lg
    mesh, P, D2 = _build_system(basis)
    N = mesh.shape[0]
    fvec = rhs(mesh)
    rng = np.random.default_rng(seed)
    A = D2.copy(); b = fvec.copy()
    if k > 0:
        if fixed_ti is not None and k == 1:
            idx = np.array([int(np.argmin(np.abs(mesh - fixed_ti)))])
        else:
            idx = rng.choice(N, size=k, replace=False)
        A = np.vstack([D2, 3.0 * P[idx]])
        b = np.concatenate([fvec, 3.0 * ref(mesh[idx])])
    c_particular, *_ = np.linalg.lstsq(A, b, rcond=None)
    _, s, vh = np.linalg.svd(A, full_matrices=True)
    rank = int(np.sum(s > 1e-8))
    null_basis = vh[rank:]
    if null_basis.shape[0] > 0:
        coeff = rng.normal(0.0, 0.5, size=null_basis.shape[0])
        c_family = c_particular + coeff @ null_basis
        v = null_basis[0]
    else:
        c_family = c_particular
        v = None
    xv = np.linspace(-1, 1, 321)
    uvals = Lg.legval(xv, np.eye(basis)).T @ c_family
    return uvals, v


def _direction_metrics(ti: float, system: int = 0, k: int = 1,
                       seeds: int = 8, basis: int = 12) -> dict:
    """k=1 约束钉在精确位置 ti 的核方向分解(全部真实数值).

    理论预言(可证伪): 残余方向 = (t - ti) ⇒ v=(c0,c1,0,...) 且
      lin_frac = c1²/(c0²+c1²) ≈ 1/(1+ti²),  rest_norm ≈ 0.
    返回: {ti, c0, c1, rest_norm, lin_frac, const_frac, sigmaH_full,
           w_const, w_lin, pred_lin_frac, err}
      w_const/w_lin: σH 按方向归因权重 —— w_const=|c0|/mean|v|, w_lin=mean|c1·t|/mean|v|.
      中心 ti=0: c0≈0 ⇒ w_lin≈1 —— "σH 完全由线性核方向承载"(复核 X8 解读).
    """
    label, rhs, ref, _refp = _SYSTEMS[system]
    evs = []
    for s in range(seeds):
        u, v = _solve_under_decomp(k, s, basis, rhs=rhs, ref=ref, fixed_ti=ti)
        evs.append(u)
    evs = np.stack(evs)
    stds = evs.std(axis=0)
    amean = np.abs(evs.mean(axis=0))
    sigmaH_full = float(stds.mean() / (amean.mean() + 1e-9))
    if k >= 2:
        return {"ti": round(ti, 4), "system": system, "k": k,
                "sigmaH_full": round(sigmaH_full, 6),
                "null_dim": 0, "lin_frac": 0.0, "const_frac": 0.0,
                "c0": 0.0, "c1": 0.0, "rest_norm": 0.0, "w_const": 0.0, "w_lin": 0.0,
                "pred_lin_frac": 0.0, "err": round(sigmaH_full, 6)}
    u, v = _solve_under_decomp(k, 0, basis, rhs=rhs, ref=ref, fixed_ti=ti)
    c0, c1 = float(v[0]), float(v[1])
    rest = float(np.linalg.norm(v[2:]))
    norm2 = c0 * c0 + c1 * c1 + 1e-12
    lin_frac = c1 * c1 / norm2
    const_frac = c0 * c0 / norm2
    t_lin = np.linspace(-1, 1, 321)
    vabs = np.abs(c0 + c1 * t_lin)
    mv = float(vabs.mean()) + 1e-12
    w_const = float(abs(c0) / mv)
    w_lin = float(np.abs(c1 * t_lin).mean() / mv)
    pred = 1.0 / (1.0 + ti * ti)
    return {"ti": round(ti, 4), "system": system, "k": k,
            "c0": round(c0, 6), "c1": round(c1, 6), "rest_norm": float(rest),
            "lin_frac": round(lin_frac, 4), "const_frac": round(const_frac, 4),
            "sigmaH_full": round(sigmaH_full, 4),
            "w_const": round(w_const, 4), "w_lin": round(w_lin, 4),
            "pred_lin_frac": round(pred, 4), "err": round(abs(lin_frac - pred), 4)}


def _solve_general(order: int, ctype: str, k: int, seed: int, basis: int,
                   system: int = 0, fixed_ti: float | None = None):
    """泛化解族求解: order 阶 ODE + ctype 约束(point 值 / derivative 导数).

    欠定 D_order u = rhs + k 个约束 → 零空间采样(同 _solve_under 语义):
      - point      : 约束行 = 3·P(t_i),      b = 3·ref(t_i)
      - derivative : 约束行 = 3·P1(t_i),     b = 3·ref'(t_i)
    order=2 核=span{1,t}(2维); order=3 核=span{1,t,t²}-Legendre 版(3维).
    返回 (uvals, null_basis); 核被完全钳制时 null_basis 为空.
    """
    from numpy.polynomial import legendre as Lg
    mesh, P, D = _build_system_g(basis, order=order)
    N = mesh.shape[0]
    fvec = _SYSTEMS[system][1](mesh)
    if ctype == "derivative":
        P1 = _build_p1(basis)
        C = P1
    else:
        C = P
    rng = np.random.default_rng(seed)
    A = D.copy(); b = fvec.copy()
    if k > 0:
        if fixed_ti is not None:
            # 受控实验: 首个约束钉在 fixed_ti, 其余 k-1 个约束随机(对 k=1 即精确钉 ti)
            idx0 = np.array([int(np.argmin(np.abs(mesh - fixed_ti)))])
            if k == 1:
                idx = idx0
            else:
                rest = rng.choice(np.setdiff1d(np.arange(N), idx0),
                                  size=k - 1, replace=False)
                idx = np.concatenate([idx0, rest]).astype(int)
        else:
            idx = rng.choice(N, size=k, replace=False)
        A = np.vstack([D, 3.0 * C[idx]])
        if ctype == "derivative":
            _refp = _SYSTEMS[system][3]
            b = np.concatenate([fvec, 3.0 * _refp(mesh[idx])])
        else:
            _ref = _SYSTEMS[system][2]
            b = np.concatenate([fvec, 3.0 * _ref(mesh[idx])])
    c_particular, *_ = np.linalg.lstsq(A, b, rcond=None)
    _, s, vh = np.linalg.svd(A, full_matrices=True)
    rank = int(np.sum(s > 1e-8))
    null_basis = vh[rank:]
    if null_basis.shape[0] > 0:
        coeff = rng.normal(0.0, 0.5, size=null_basis.shape[0])
        c_family = c_particular + coeff @ null_basis
    else:
        c_family = c_particular
    xv = np.linspace(-1, 1, 321)
    uvals = Lg.legval(xv, np.eye(basis)).T @ c_family
    return uvals, null_basis


def _direction_metrics_g(order: int, ctype: str, ti: float, system: int = 0,
                         k: int = 1, seeds: int = 8, basis: int = 12) -> dict:
    """泛化核方向分解(第 9 轮+ 白名单新维度).

    残余零空间方向投到前 order 个 Legendre 核系数 {P0,P1,...}, 归一化得权重:
      dir_weights[j] = 平均方向在 P_j 上的占比.
    类型预言(可证伪, 已数值验证):
      (2, point)      : lin_frac(=w1) = 1/(1+t_i²)              —— 值约束位置定律
      (2, derivative) : dir ≈ (1,0) 纯常数核, **位置无关**       —— 导数约束钉线性核
      (3, point)      : null_dim = 2 (order-k), 残余张成 P1/P2 面
      (3, derivative) : null_dim = 2, 线性核在 ti=0 被钉(w1≈0)
    返回统一字段: {order, ctype, ti, system, k, null_dim, dir_weights,
                   const_frac, lin_frac, quad_frac, sigmaH_full, pred, err}
      pred/err 按类型取判别量: (2,point)=lin_frac vs 1/(1+t_i²);
      (2,deriv)=lin_frac vs 0; (3,*)=null_dim vs 2.
    """
    evs = []
    for s in range(seeds):
        u, nb = _solve_general(order, ctype, k, s, basis, system=system, fixed_ti=ti)
        evs.append(u)
    evs = np.stack(evs)
    sigmaH_full = float(evs.std(axis=0).mean() / (np.abs(evs).mean() + 1e-9))
    u, nb = _solve_general(order, ctype, k, 0, basis, system=system, fixed_ti=ti)
    null_dim = int(nb.shape[0]) if nb is not None else 0
    weights = np.zeros(order)
    if null_dim > 0:
        for v in nb[:(order)]:                # 最多取 order 个独立方向(秩 ≤ order-k)
            w = v[:order] ** 2
            s_w = w.sum() + 1e-12
            weights += w / s_w
        weights /= min(null_dim, order)       # 平均方向权重(归一化)
    const_frac = float(weights[0]) if order >= 1 else 0.0
    lin_frac = float(weights[1]) if order >= 2 else 0.0
    quad_frac = float(weights[2]) if order >= 3 else 0.0
    # 判据按 (order, ctype, k): 定律只在 k=1 成立; k>=2 退化到"完全钳制"签名.
    # 导数约束的实质: 约束行 = P'(t_i), 而 P0'=0 ⇒ 常数核永不被导数约束钳制
    #   ⇒ 纯导数约束的残余 null_dim 恒 ≥ 1(与值约束的"可钳到 0"形成本质对照).
    if (order, ctype) == (2, "point") and k == 1:
        pred = 1.0 / (1.0 + ti * ti); err = float(abs(lin_frac - pred))
    elif ctype == "derivative":
        pred = float(max(1, order - k)); err = float(abs(null_dim - pred))
    else:                                      # point 约束(含 k>=2 完全钳制)
        pred = float(max(0, order - k)); err = float(abs(null_dim - pred))
    return {"order": order, "ctype": ctype, "ti": round(ti, 4), "system": system,
            "k": k, "null_dim": null_dim,
            "dir_weights": [round(float(x), 4) for x in weights],
            "const_frac": round(const_frac, 4), "lin_frac": round(lin_frac, 4),
            "quad_frac": round(quad_frac, 4),
            "sigmaH_full": round(sigmaH_full, 4), "pred": round(pred, 4),
            "err": round(err, 4)}


def exp_k1_direction(seeds: int = 8, basis: int = 12) -> dict:
    """X9 · 核方向分解: k=1 约束钉中心 vs 边界, 显式分解残余零空间方向.

    直接回答报告'下一步'第2条与第3条的证据部分:
      - 中心 ti≈0: v≈(0,1) ⇒ lin_frac≈1, w_lin≈1 —— 残余是**纯线性核**, 
        修正 X8 '中心几乎测不到线性核'的归一化误导(那是 |v(x)|=|x| 幅值小, 非方向消失);
      - 边界 ti≈0.97: v≈(-0.97,1) ⇒ lin_frac≈0.52 —— 常数+线性混合.
    客观: 方向分解下的中心-边界「线性核占比」差(中心 1 vs 边界 0.52)最大化.
    """
    c = _direction_metrics(0.0, system=0, k=1, seeds=seeds, basis=basis)
    b = _direction_metrics(0.97, system=0, k=1, seeds=seeds, basis=basis)
    purity = float(c["lin_frac"])            # 中心应≈1(纯线性核)
    return {
        "objectives": {"k1_direction_purity": purity},
        "summary": {"seeds": seeds, "basis": basis,
                    "center": c, "boundary": b,
                    "k1_direction_purity": round(purity, 4),
                    "correction_note": ("X8 '中心测不到线性核'为归一化伪影: 中心 v=(0,1) 纯线性核, "
                                        "σH 低是因 |t| 幅值小, 非方向消失; 边界 v 为常数+线性混合.")},
        "success": True,
    }


def exp_position_curve(seeds: int = 8, basis: int = 12,
                       positions=(-0.97, -0.66, -0.33, 0.0, 0.33, 0.66, 0.97)) -> dict:
    """X10 · 位置→方向曲线: 扫约束位置, 检验方向定律 lin_frac(t_i)=1/(1+t_i²).

    同一系统(cos)下扫 ti, 每条 ODE 真实求解+零空间分解 —— 定量曲线.
    客观: 定律拟合分 1/(1+mse) 最大化(与预言偏差越小越好).
    """
    rows = []
    for ti in positions:
        d = _direction_metrics(ti, system=0, k=1, seeds=seeds, basis=basis)
        rows.append({k: d[k] for k in ("ti", "lin_frac", "pred_lin_frac", "err",
                                       "w_const", "w_lin", "sigmaH_full")})
    mse = float(np.mean([r["err"] ** 2 for r in rows]))
    fit = float(1.0 / (1.0 + mse))
    return {
        "objectives": {"locality_law_fit": fit},
        "summary": {"seeds": seeds, "basis": basis, "positions": list(positions),
                    "rows": rows, "mse": round(mse, 6),
                    "law": "lin_frac(t_i) = 1/(1+t_i^2)",
                    "locality_law_fit": round(fit, 4)},
        "success": True,
    }


def exp_direction_universal(seeds: int = 8, basis: int = 12) -> dict:
    """X11 · 方向定律普适: 三系统下中心/边界的 lin_frac 是否同服从 1/(1+t_i²).

    齐次核 span{1,t} 对所有 u''=rhs 恒成立 ⇒ 方向定律应跨系统不变(强可证伪).
    客观: 普适吻合度 = 1/(1+max|measured-pred|) 最大化.
    """
    per = []
    for si in range(3):
        c = _direction_metrics(0.0, system=si, k=1, seeds=seeds, basis=basis)
        b = _direction_metrics(0.97, system=si, k=1, seeds=seeds, basis=basis)
        per.append({"system": _SYSTEMS[si][0], "center_lin_frac": c["lin_frac"],
                    "boundary_lin_frac": b["lin_frac"],
                    "pred_boundary": b["pred_lin_frac"],
                    "ci_center": c["err"], "err_boundary": b["err"]})
    max_err = max(r["err_boundary"] + r["ci_center"] for r in per)
    univ = float(1.0 / (1.0 + max_err))
    return {
        "objectives": {"direction_universality": univ},
        "summary": {"seeds": seeds, "basis": basis, "systems": per,
                    "max_err": round(max_err, 6),
                    "direction_universality": round(univ, 4)},
        "success": True,
    }


def _sanitize_scan(cfg: dict | None) -> dict:
    """把书生提议的扫描配置校验/落入白名单(越界回落到默认, 绝不执行编造的实验)."""
    out = dict(_SCAN_DEFAULTS)
    try:
        cfg = cfg or {}
        if cfg.get("system") in SCAN_OPS["system"]:
            out["system"] = int(cfg["system"])
        if cfg.get("k") in SCAN_OPS["k"]:
            out["k"] = int(cfg["k"])
        ps = [float(p) for p in (cfg.get("positions") or [])]
        ps = [p for p in ps if p in SCAN_OPS["positions"]]
        if ps:
            out["positions"] = list(dict.fromkeys(ps))
        if cfg.get("basis") in SCAN_OPS["basis"]:
            out["basis"] = int(cfg["basis"])
        if cfg.get("seeds") in SCAN_OPS["seeds"]:
            out["seeds"] = int(cfg["seeds"])
        if cfg.get("order") in SCAN_OPS["order"]:
            out["order"] = int(cfg["order"])
        if cfg.get("ctype") in SCAN_OPS["ctype"]:
            out["ctype"] = str(cfg["ctype"])
    except Exception:  # noqa: BLE001 — 无法解析的配置一律用默认(诚实回退)
        pass
    if out["positions"][0] != min(out["positions"]):
        out["positions"].sort()
    return out


def exp_dir_scan(cfg: dict, name: str) -> dict:
    """X12+ · 通用方向扫描: 书生在第 3+ 轮提议的 (order, ctype, system, k, positions, basis, seeds).

    每个位置真实求解(任意阶 ODE)+核方向分解; 双目标(独立维度):
      scan_fit   : 1/(1+mse vs 该配置类型的解析预言) —— 机制签名是否成立;
      scan_extent: 残余方向偏好 —— (2,*) 取平均 lin_frac; (3,*) 取平均 quad_frac(高次核占比).
    """
    cfg = _sanitize_scan(cfg)
    ti_map = {"0.0": 0.0, "0.33": 0.33, "0.66": 0.66, "0.97": 0.97,
              "-0.33": -0.33, "-0.66": -0.66, "-0.97": -0.97}
    rows = []
    for ti in [ti_map[str(p)] for p in cfg["positions"]]:
        d = _direction_metrics_g(order=cfg["order"], ctype=cfg["ctype"], ti=ti,
                                 system=cfg["system"], k=cfg["k"],
                                 seeds=cfg["seeds"], basis=cfg["basis"])
        rows.append(d)
    errs = [r["err"] for r in rows]
    mse = float(np.mean([e * e for e in errs]))
    fit = float(1.0 / (1.0 + mse))
    if cfg["order"] == 3:
        extent = float(np.mean([r["quad_frac"] for r in rows]))
    else:
        lins = [r["lin_frac"] for r in rows if r["k"] == 1]
        extent = float(np.mean(lins)) if lins else 0.0
    # objective 键按分支指纹化(order_ctype_k_system): 扫描分支是独立证据面,
    # 不共享键 → 互不支配 → 全部存活(保住多配置证据); 键自身含类型签名可审计.
    sig = f"o{cfg['order']}_{cfg['ctype'][:3]}_{cfg['k']}_s{cfg['system']}"
    return {
        "objectives": {f"scan_fit_{sig}": fit, f"scan_extent_{sig}": extent},
        "summary": {"name": name, **cfg, "predictions_hint": _scan_pred_hint(cfg),
                    "rows": rows,
                    "scan_fit": round(fit, 4), "scan_extent": round(extent, 4)},
        "success": True,
    }


def _scan_pred_hint(cfg: dict) -> str:
    """该配置类型的解析预言(给成文阶段模型作判读锚点, 真实可证伪)."""
    o, c = cfg["order"], cfg["ctype"]
    if (o, c) == (2, "point"):
        return f"lin_frac(t_i)=1/(1+t_i²)(值约束位置定律)"
    if c == "derivative":
        return ("导数约束钳制线性核、永不钳制常数核(P0'=0) ⇒ 残余含常数方向, "
                "null_dim 恒 ≥1")


def _make_experiments(cycle: int = 1):
    """按轮次返回真实证据分支 → Experiment 列表.

    cycle=1: X1..X8(判据/预算/约束曲线/普适四方向);
    cycle=2: X9(核方向分解) + X10(位置-方向定律) + X11(方向定律跨系统普适);
    cycle>=3: 由书生提议配置(X12+ 通用方向扫描, 见 _make_scan_experiments).
    """
    from huginn.research import Experiment

    if cycle == 1:
        specs = [
            (lambda: exp_eps_criterion(), "X1_eps_criterion",
             "eps判据(基准): 固定N扫eps, 刚性cos的C*在阈值后平台化而胖|x|持续增长 → "
             "饱和由精度截止eps的判据区分度拉大两类系统."),
            (lambda: exp_n_scale(), "X4_n_platform",
             "N平台(对照组): 固定eps扫N=512..8192, 刚/胖的C*对N平台化 → 饱和判据不依赖N."),
            (lambda: exp_constraint_curve(), "X5_constraint_curve",
             "约束全曲线: k=0..3 的sigmaH序列单调下降 → 约束数逐条钳制解族核自由度, "
             "建立「约束数→解族维数」定量曲线."),
            (lambda: exp_budget_sweep(), "X6_budget_sweep",
             "预算扫描: 训练预算递增下vho收敛曲线 → 量化优化器财政的假饱和窗口."),
            (lambda: exp_multi_system(), "X7_multi_system",
             "多系统普适(修 ref): 修正第三个系统 ref=t^4/12-t^2/2 后, 若 σH_k2 全系统归零 "
             "则上一轮'多项式系统减弱'是实验定义伪影, 约束钳制跨系统普适."),
            (lambda: exp_k1_locality(), "X8_k1_locality",
             "k=1 中间态位置依赖: 约束钉在中心(测不到线性核)vs 边界(钳住线性核)的 σH_k1 差 → "
             "1 个点值约束钳制哪个核方向由位置决定."),
            (lambda: _exp_optimizer_finance(), "X2_optimizer_finance",
             "优化器财政(确认): 真实PINN在固定训练预算下的留出误差vho若未收敛, "
             "容量饱和信号会被预算不足的'假饱和'污染(判据须加财政守卫)."),
            (lambda: exp_constraint_dimension(), "X3_constraint_dimension",
             "约束维数(确认): 欠定u''=-cos + k个点值约束 → k=0 vs k=2 的sigmaH钳制效应."),
        ]
        return [Experiment(name=n, hypothesis=h, run=fn) for fn, n, h in specs]

    if cycle == 2:
        specs = [
            (lambda: exp_k1_direction(), "X9_k1_direction",
             "核方向分解: k=1 约束钉中心(ti≈0)=>lin_frac≈1 残余纯线性核, 边界(ti≈0.97)"
             "=>常数+线性混合 —— 用零空间方向分解复核 X8 '中心测不到线性核'是否归一化伪影."),
            (lambda: exp_position_curve(), "X10_position_curve",
             "位置-方向定律: 扫 ti∈±{0.33,0.66,0.97,0.0}, lin_frac 应服从 1/(1+t_i²) "
             "→ 建立约束位置→核方向占比的定量曲线."),
            (lambda: exp_direction_universal(), "X11_direction_universal",
             "方向定律普适: 三系统(cos/sin2t/多项式)下中心/边界 lin_frac 同服 1/(1+t_i²)"
             "→ 方向钳制机制跨系统普适(齐次核 span{1,t} 对任意 u''=rhs 成立)."),
        ]
        return [Experiment(name=n, hypothesis=h, run=fn) for fn, n, h in specs]

    raise ValueError(f"cycle={cycle} 无内置实验; cycle>=3 应走 _make_scan_experiments")


def _make_scan_experiments(configs: list[dict]) -> list:
    """第 3+ 轮: 书生提议的扫描配置 → 真实 Experiment 分支(白名单校验).

    每个配置一个独立分支, 双目标(scan_fit 定律拟合 / scan_extent 方向倾向),
    互相独立不支配 → 全部存活, 保住多配置证据面.
    """
    from huginn.research import Experiment
    exps = []
    for i, cfg in enumerate(configs):
        c = _sanitize_scan(cfg)
        name = f"S{i + 1}_scan"
        exps.append(Experiment(
            name=name,
            hypothesis=(f"方向扫描 config #{i + 1}: system={c['system']} k={c['k']} "
                        f"positions={c['positions']} basis={c['basis']} seeds={c['seeds']} "
                        f"order={c['order']} ctype={c['ctype']} "
                        f"→ 检验核方向机制签名 {_scan_pred_hint(c)} (书生本轮提议)."),
            run=lambda cc=c, nn=name: exp_dir_scan(cc, nn)))
    return exps


def _build_plan(goal: str, run_by_name: dict, cycle: int = 1):
    """需求拆解 -> 带依赖 DAG -> 分层(按轮次).

    cycle=1 层0: X1 (判据基准);  层1: X4/N平台 + X5/约束曲线 + X6/预算扫描;
           层2: X7/多系统 + X8/k1位置依赖 + X2/X3 (确认组).
    cycle=2 层0: X10/位置-方向定律 (基准曲线);
           层1: X9/核方向分解 + X11/方向律普适 (依赖曲线, 并行).
    cycle>=3: 扫描分支无层间依赖 → 单层全并行(书生提议的配置彼此独立).
    """
    from huginn.research.planning import build_research_plan, SubResearch
    if cycle == 1:
        return build_research_plan(goal, [
            SubResearch("X1_eps_criterion", "基准判据: eps 区分刚性/胖",
                        run=run_by_name["X1_eps_criterion"], depends_on=[]),
            SubResearch("X4_n_platform", "N平台对照: C* 对 N 不敏感",
                        run=run_by_name["X4_n_platform"], depends_on=["X1_eps_criterion"]),
            SubResearch("X5_constraint_curve", "约束全曲线 k=0..3",
                        run=run_by_name["X5_constraint_curve"], depends_on=["X1_eps_criterion"]),
            SubResearch("X6_budget_sweep", "预算扫描: 假饱和窗口",
                        run=run_by_name["X6_budget_sweep"], depends_on=["X1_eps_criterion"]),
            SubResearch("X7_multi_system", "多系统普适(修 ref)",
                        run=run_by_name["X7_multi_system"],
                        depends_on=["X5_constraint_curve"]),
            SubResearch("X8_k1_locality", "k=1 中间态位置依赖",
                        run=run_by_name["X8_k1_locality"],
                        depends_on=["X5_constraint_curve"]),
            SubResearch("X2_optimizer_finance", "优化器财政确认",
                        run=run_by_name["X2_optimizer_finance"],
                        depends_on=["X1_eps_criterion"]),
            SubResearch("X3_constraint_dimension", "约束维数确认",
                        run=run_by_name["X3_constraint_dimension"],
                        depends_on=["X5_constraint_curve"]),
        ], parallel_cap=3)
    if cycle == 2:
        return build_research_plan(goal, [
            SubResearch("X10_position_curve", "位置-方向定律基准曲线",
                        run=run_by_name["X10_position_curve"], depends_on=[]),
            SubResearch("X9_k1_direction", "核方向分解(中心vs边界)",
                        run=run_by_name["X9_k1_direction"],
                        depends_on=["X10_position_curve"]),
            SubResearch("X11_direction_universal", "方向定律跨系统普适",
                        run=run_by_name["X11_direction_universal"],
                        depends_on=["X10_position_curve"]),
        ], parallel_cap=3)
    names = list(run_by_name.keys())
    return build_research_plan(goal, [
        SubResearch(n, f"书生提议扫描分支 {n}", run=run_by_name[n], depends_on=[])
        for n in names
    ], parallel_cap=3)


def _diagnostic_tools():
    """书生在成文阶段可自主调用的域诊断工具 (返回真实数值, 进 trace 供门禁核对)."""
    def h_probe_cstar(a):
        target = a.get("target", "rigid")
        N = int(a.get("N", 4096))
        eps_list = a.get("eps_list") or [1e-2, 1e-4, 1e-6, 1e-8]
        return json.dumps({"target": target, "Cstar": {
            "eps": eps_list, "value": [_poly_cstar(target, e, N) for e in eps_list]}},
            ensure_ascii=False)

    def h_probe_optimizer(a):
        res = _exp_optimizer_finance(int(a.get("w", 32)), int(a.get("steps", 1600)))
        return json.dumps(res["summary"], ensure_ascii=False)

    def h_probe_constraint(a):
        seeds = int(a.get("seeds", 8))
        evs = []
        for k in (0, 1, 2):
            vals = np.stack([_solve_under(k, s, 12) for s in range(seeds)])
            sig = float(vals.std(axis=0).mean() / (np.abs(vals).mean() + 1e-9))
            evs.append({"k": k, "sigmaH": round(sig, 4)})
        return json.dumps({"sigmaH_by_k": evs}, ensure_ascii=False)

    def h_probe_n_platform(a):
        eps = float(a.get("eps", 1e-4))
        Ns = a.get("Ns") or [512, 1024, 2048, 4096, 8192]
        rows = []
        for N in Ns:
            rows.append({"N": N,
                         "Cstar_rigid": _poly_cstar("rigid", eps, N),
                         "Cstar_fat": _poly_cstar("fat", eps, N)})
        return json.dumps({"eps": eps, "rows": rows}, ensure_ascii=False)

    def h_probe_k1_locality(a):
        seeds = int(a.get("seeds", 8))
        center = float(a.get("center", 0.0))
        boundary = float(a.get("boundary", 0.97))
        def _sig(ti):
            evs = np.stack([_solve_under(1, s, 12, rhs=lambda t: -np.cos(t),
                                         ref=lambda t: np.cos(t), fixed_ti=ti)
                            for s in range(seeds)])
            std = float(evs.std(axis=0).mean()); mean = float(np.abs(evs).mean())
            return std, mean, float(std / (mean + 1e-9))
        cs, cm, cr = _sig(center); bs, bm, br = _sig(boundary)
        return json.dumps({"center": {"abs_std": round(cs, 4), "abs_mean": round(cm, 4),
                                      "sigmaH": round(cr, 4)},
                           "boundary": {"abs_std": round(bs, 4), "abs_mean": round(bm, 4),
                                        "sigmaH": round(br, 4)},
                           "locality_norm": round(cr - br, 4),
                           "locality_abs": round(cs - bs, 4)}, ensure_ascii=False)

    def h_probe_kernel_direction(a):
        """k=1 约束钉 ti 的零空间方向分解: c0/c1/lin_frac/sigmaH(真实)."""
        ti = float(a.get("ti", 0.0))
        return json.dumps(_direction_metrics(ti, system=int(a.get("system", 0)),
                                             k=int(a.get("k", 1)),
                                             seeds=int(a.get("seeds", 8)),
                                             basis=int(a.get("basis", 12))),
                          ensure_ascii=False)

    def h_probe_direction_curve(a):
        """扫约束位置, 返回 lin_frac(t_i) 与预言 1/(1+t_i²) 的曲线对照(真实)."""
        positions = a.get("positions") or [0.0, 0.33, 0.66, 0.97]
        rows = []
        for ti in positions:
            d = _direction_metrics(float(ti), system=int(a.get("system", 0)),
                                   k=int(a.get("k", 1)),
                                   seeds=int(a.get("seeds", 8)),
                                   basis=int(a.get("basis", 12)))
            rows.append({"ti": d["ti"], "lin_frac": d["lin_frac"],
                         "pred_lin_frac": d["pred_lin_frac"], "err": d["err"],
                         "sigmaH_full": d["sigmaH_full"]})
        return json.dumps({"rows": rows}, ensure_ascii=False)

    def h_probe_direction_g(a):
        """泛化核方向探针: 任意 order(2=低次核/3=高次核) × ctype(point值/derivative导数)."""
        rows = []
        for ti in (a.get("positions") or [0.0, 0.97]):
            d = _direction_metrics_g(order=int(a.get("order", 2)),
                                     ctype=str(a.get("ctype", "point")),
                                     ti=float(ti), system=int(a.get("system", 0)),
                                     k=int(a.get("k", 1)),
                                     seeds=int(a.get("seeds", 8)),
                                     basis=int(a.get("basis", 12)))
            rows.append({kk: d[kk] for kk in ("ti", "order", "ctype", "null_dim",
                                              "dir_weights", "const_frac", "lin_frac",
                                              "quad_frac", "sigmaH_full", "pred", "err")})
        return json.dumps({"rows": rows}, ensure_ascii=False)

    return [
        {"tool": {"function": {"name": "probe_Cstar",
            "description": "固定N扫eps, 返回刚性/胖系统达到精度eps所需最小容量C*曲线(真实)",
            "parameters": {"type": "object", "properties": {
                "target": {"type": "string", "enum": ["rigid", "fat"]},
                "eps_list": {"type": "array", "items": {"type": "number"}},
                "N": {"type": "integer"}}, "required": ["target"],
                "additionalProperties": False}}}, "handle": h_probe_cstar},
        {"tool": {"function": {"name": "probe_n_platform",
            "description": "固定eps扫N, 返回刚/胖C*随N的平台化对照(N平台探针, 检验判据是否依赖N)",
            "parameters": {"type": "object", "properties": {
                "eps": {"type": "number"},
                "Ns": {"type": "array", "items": {"type": "integer"}}},
                "required": [], "additionalProperties": False}}}, "handle": h_probe_n_platform},
        {"tool": {"function": {"name": "probe_k1_locality",
            "description": "k=1中间态: 单点值约束钉在中心(测不到线性核)vs边界(钳住)的sigmaH_k1差(位置依赖探针)",
            "parameters": {"type": "object", "properties": {
                "seeds": {"type": "integer"}, "center": {"type": "number"},
                "boundary": {"type": "number"}},
                "required": [], "additionalProperties": False}}}, "handle": h_probe_k1_locality},
        {"tool": {"function": {"name": "probe_kernel_direction",
            "description": "k=1约束钉在位置ti的零空间方向分解(c0/c1/lin_frac/const_frac/sigmaH): 残余方向落在哪个核",
            "parameters": {"type": "object", "properties": {
                "ti": {"type": "number"}, "system": {"type": "integer"},
                "k": {"type": "integer"}, "seeds": {"type": "integer"},
                "basis": {"type": "integer"}},
                "required": [], "additionalProperties": False}}}, "handle": h_probe_kernel_direction},
        {"tool": {"function": {"name": "probe_direction_curve",
            "description": "扫约束位置, 返回lin_frac(t_i)与预言1/(1+t_i^2)曲线对照(核方向定律检验)",
            "parameters": {"type": "object", "properties": {
                "positions": {"type": "array", "items": {"type": "number"}},
                "system": {"type": "integer"}, "k": {"type": "integer"},
                "seeds": {"type": "integer"}, "basis": {"type": "integer"}},
                "required": [], "additionalProperties": False}}}, "handle": h_probe_direction_curve},
        {"tool": {"function": {"name": "probe_direction_g",
            "description": "泛化核方向探针: order∈{2,3}(2=低次核span{1,t} / 3=高次核span{1,t,t^2}) × "
                           "ctype∈{point,derivative}(值约束 u(t_i) / 导数约束 u'(t_i)) → 核方向权重/维度签名",
            "parameters": {"type": "object", "properties": {
                "order": {"type": "integer"}, "ctype": {"type": "string"},
                "positions": {"type": "array", "items": {"type": "number"}},
                "system": {"type": "integer"}, "k": {"type": "integer"},
                "seeds": {"type": "integer"}, "basis": {"type": "integer"}},
                "required": [], "additionalProperties": False}}}, "handle": h_probe_direction_g},
        {"tool": {"function": {"name": "probe_optimizer",
            "description": "真实PINN在给定宽度w与训练预算steps下的训练残差/留出误差vho(优化器财政审计)",
            "parameters": {"type": "object", "properties": {
                "w": {"type": "integer"}, "steps": {"type": "integer"}},
                "required": [], "additionalProperties": False}}}, "handle": h_probe_optimizer},
        {"tool": {"function": {"name": "probe_constraint_dim",
            "description": "对欠定u''=-cos, 用不同种子数重算k=0/1/2约束下的解族重解散布sigmaH(解族维数探针)",
            "parameters": {"type": "object", "properties": {"seeds": {"type": "integer"}},
                "required": [], "additionalProperties": False}}}, "handle": h_probe_constraint},
    ]


def _extract_next_open(report_text: str) -> str:
    """从上一轮报告里提取「下一步/局限」开放问题(供下一轮 goal 拼接).

    兼容编号标题(如 "## 5. 下一步")与无编号标题; 提取到首个后续标题前为止.
    """
    pat = re.compile(r"#+\s*(?:[0-9]+[.、)]?\s*)*(下一步|局限|后续工作|未来工作)"
                     r"[^\n]*\n(.*?)(?=\n[#]{1,3}\s|\Z)", flags=re.DOTALL)
    m = pat.search(report_text or "")
    if not m:
        # 兜底: 最后 900 字符(结论段) —— 至少保留"本轮收尾视角"进下轮 context
        return (report_text or "").strip()[-900:]
    return m.group(2).strip()[:2000]


def _ask_json(client, model: str, system: str, user: str, max_tokens: int = 900) -> dict:
    """让书生输出 JSON(解析失败/空返回 {}, 绝不阻塞主线)."""
    try:
        r = client.chat.completions.create(
            model=model, max_tokens=max_tokens, temperature=0.2,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}])
        text = r.choices[0].message.content or ""
    except Exception as ee:  # noqa: BLE001 — 提议失败不阻断, 调用方走默认
        print(f"  [提议] LLM 失败: {ee}")
        return {}
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return {}


def _propose_next_open(client, model: str, report_text: str) -> list[str]:
    """书生[观察]: 读上一轮报告, 提出下一轮最重要的开放问题(3 条以内)."""
    d = _ask_json(
        client, model,
        "你是长程科研规划者。基于前一周期报告, 提出下一周期最值得攻克的开放问题。"
        "要求: 问题必须能被 {零空间方向分解:{system∈[0,1,2], k∈[1,2], positions∈"
        "[-0.97,-0.66,-0.33,0,0.33,0.66,0.97], basis∈[8,12,16], seeds∈[4,8,12]}} "
        "这类真实数值实验检验; 每条一句, 指向具体可证伪预言。只输出 JSON, 格式: "
        '{"open_questions": ["q1", "q2"]}, 不含其他文字。',
        f"上一周期报告:\n{report_text[:6000]}")
    qs = [str(q).strip() for q in (d.get("open_questions") or []) if str(q).strip()]
    return qs[:3]


def _propose_scan_configs(client, model: str, next_open: str,
                          done_cfgs: list[dict] | None = None) -> list[dict]:
    """书生[行动规划]: 基于下一轮开放问题, 在白名单内提议 ≤3 个扫描配置.

    已执行过的配置指纹(done_cfgs)进 prompt —— 强制避开重复, 把预算投向
    开放问题指出的未检维度(如 k=2 双约束、ti=±1.0、不同 system/basis 组合).
    """
    done = " ".join(json.dumps(c, sort_keys=True, ensure_ascii=False)
                    for c in (done_cfgs or [])) or "无"
    d = _ask_json(
        client, model,
        "你是实验设计者。基于给定的开放问题, 从白名单 {system∈{0,1,2}(0=cos, 1=sin2t, "
        "2=多项式 u''=-(1-t^2)), k∈{1,2}, positions∈[-0.97,-0.66,-0.33,0,0.33,0.66,0.97], "
        "basis∈{8,12,16}, seeds∈{4,8,12}, order∈{2,3}(2=低次核 span{1,t}; 3=高次核 "
        "span{1,t,t²}), ctype∈{point,derivative}(值约束 u(t_i) / 导数约束 u'(t_i))} 里设计"
        "最多3个互不重复的真实数值实验. 必须避开已经执行过的配置(它们不再提供新信息), "
        "优先选择能直接检验开放问题的未做配置(如开放问题提到'约束数量', 就设计 k=2 且 "
        "positions 含多个位置的实验; 提到'高阶/高次/三阶/核维', 就设计 order=3 且 positions "
        "含 0.0 与 0.97; 提到'导数/梯度/混合约束', 就设计 ctype=derivative; 提到'系统普适', "
        "就覆盖三个 system). 已执行配置:\n" + done + "\n只输出 JSON: {\"configs\": "
        '[{"system":0,"k":1,"positions":[0.0,0.97],"basis":12,"seeds":8,'
        '"order":2,"ctype":"point"}], 不含其他文字。}',
        f"下一轮开放问题:\n{next_open[:2500]}")
    cfgs = d.get("configs") or []
    ok = []
    for c in cfgs[:3]:
        if isinstance(c, dict):
            ok.append(c)
    return ok


def _pad_scan_configs(cfgs: list[dict], open_text: str) -> list[dict]:
    """兜底/补齐: 保证 ≥3 个配置, 并优先覆盖开放问题点名的未检维度.

    覆盖策略(全部真实, 不编造): 若开放问题提到约束数量 → 补 k=2 配置;
    提到高阶/高次/三阶/核维 → 补 order=3 配置; 提到导数/梯度 → 补 ctype=derivative;
    提到边界/ti=1.0/位置 → 补 ±0.97·0.0 组合; 提到系统/普适 → 补 system 0/1/2 横扫.
    """
    out = list(cfgs)
    text = open_text or ""
    wants_k2 = ("约束数量" in text or "k=2" in text or "k=0" in text
                or "约束阶" in text or "对照" in text)
    wants_order3 = ("高阶" in text or "高次" in text or "三阶" in text
                    or "核维" in text or "order" in text or "高阶核" in text)
    wants_deriv = ("导数" in text or "梯度" in text or "混合约束" in text)
    wants_boundary = ("边界" in text or "ti=1.0" in text or "位置" in text
                      or "曲线" in text)
    wants_sys = ("系统" in text or "普适" in text)
    cand = []
    if wants_k2:
        cand.append({"system": 0, "k": 2, "positions": [0.0, 0.33, 0.66, 0.97],
                     "basis": 12, "seeds": 8})
        cand.append({"system": 1, "k": 2, "positions": [0.0, 0.97],
                     "basis": 12, "seeds": 8})
    if wants_order3:
        cand.append({"system": 0, "k": 1, "positions": [0.0, 0.33, 0.66, 0.97],
                     "basis": 12, "seeds": 8, "order": 3})
        cand.append({"system": 2, "k": 2, "positions": [0.0, 0.97],
                     "basis": 12, "seeds": 8, "order": 3})
    if wants_deriv:
        cand.append({"system": 0, "k": 1, "positions": [0.0, 0.66, 0.97],
                     "basis": 12, "seeds": 8, "ctype": "derivative"})
        cand.append({"system": 1, "k": 1, "positions": [0.0, 0.97],
                     "basis": 12, "seeds": 8, "ctype": "derivative"})
    if wants_boundary:
        cand.append({"system": 0, "k": 1,
                     "positions": [-0.97, -0.66, -0.33, 0.0, 0.33, 0.66, 0.97],
                     "basis": 12, "seeds": 8})
    if wants_sys:
        for si in range(3):
            cand.append({"system": si, "k": 1, "positions": [0.0, 0.97],
                         "basis": 12, "seeds": 8})
    cand.append({"system": 2, "k": 1, "positions": [0.0, 0.33, 0.66, 0.97],
                 "basis": 12, "seeds": 8})
    cand.append({"system": 0, "k": 1, "positions": [0.0, 0.97],
                 "basis": 12, "seeds": 8, "order": 3, "ctype": "derivative"})
    seen = set()
    for c in list(out) + cand:
        key = json.dumps(_sanitize_scan(c), sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(c)
        if len(out) >= 3:
            break
    return out[:3]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry", action="store_true", help="确定性运行(不调模型), 验证整条管线")
    ap.add_argument("--model", default="intern-s2-preview")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--max-iters", type=int, default=12)
    ap.add_argument("--min-iters", type=int, default=3)
    ap.add_argument("--strictness", type=int, default=0, choices=[0, 1, 2],
                    help="判断层·分级护栏 0/1/2(默认0=纯真伪硬门禁, 信任模型)")
    ap.add_argument("--cycles", type=int, default=2,
                    help="长程轮数: 每轮=报告→书生提开放问题→新实验→新报告, 不停顿")
    ap.add_argument("--start-cycle", type=int, default=2,
                    help="起始轮次(默认2=复用已有 cycle1 报告继续; 1=重跑探测轮)")
    args = ap.parse_args()

    from huginn.research import run_research_program, grounding_verifier
    from huginn.research import Experiment   # noqa: F401 — re-export 校验

    client = None
    if not args.dry:
        key = os.environ.get("INTERNLM_API_KEY")
        if not key:
            print("error: INTERNLM_API_KEY not set (或 --dry)", file=sys.stderr); return 2
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url=args.base_url or
                        os.environ.get("INTERNLM_BASE_URL",
                                       "https://chat.intern-ai.org.cn/api/v1"))

    _REPORT_BY_CYCLE = {1: "research_report.md", 2: "cycle2_report.md"}
    def _report_for(cycle: int) -> Path:
        return _OUT / _REPORT_BY_CYCLE.get(cycle, f"cycle{cycle}_report.md")

    print("== 书生 + Huginn 长程深研管线 ==")

    last_report = ""
    if args.start_cycle >= 2:
        prev = _report_for(1)
        if prev.exists():
            last_report = prev.read_text(encoding="utf-8")

    done_cfgs: list[dict] = []     # 已执行过的扫描配置指纹(防长程重复)

    for cycle in range(max(1, args.start_cycle), args.start_cycle + args.cycles):
        print(f"\n===== 第 {cycle} 轮(长程自主) =====")
        # ── 书生[推理·规划]: 本轮 goal 与实验集 ─────────────────────────
        if cycle == 1:
            goal = GOAL
            exps = _make_experiments(1)
            objectives = dict(_OBJECTIVES)
        elif cycle == 2:
            goal = GOAL_CYCLE2
            exps = _make_experiments(2)
            objectives = dict(_OBJECTIVES_CYCLE2)
        else:
            next_open = _extract_next_open(last_report)
            if client is not None:
                qs = _propose_next_open(client, args.model, last_report)
                next_open = " ".join(x for x in (next_open, " ".join(qs)) if x)
                if qs:
                    print(f"  [书生·观察] 下一轮开放问题: {qs}")
            if client is not None:
                cand = _propose_scan_configs(client, args.model, next_open, done_cfgs)
            else:
                cand = []
            cfgs = _pad_scan_configs(cand, next_open)  # 兜底+补齐未检维度(全为真实实验)
            fresh = [c for c in cfgs
                     if json.dumps(_sanitize_scan(c), sort_keys=True)
                     not in {json.dumps(_sanitize_scan(d), sort_keys=True)
                             for d in done_cfgs}]
            need = max(0, 3 - len(fresh))
            cfgs = fresh + [c for c in cfgs if c not in fresh][:need]
            for c in cfgs:
                done_cfgs.append(_sanitize_scan(c))
            print(f"  [书生·行动] 本轮扫描配置: {cfgs}")
            goal = (f"检验书生本轮提出的开放问题(数值证据由方向扫描分支提供): {next_open}")
            exps = _make_scan_experiments(cfgs)
            # 分支指纹化 objective 键(与 exp_dir_scan 返回值一一对应, 全 maximize).
            objectives = {}
            for c in (_sanitize_scan(c) for c in cfgs):
                sig = f"o{c['order']}_{c['ctype'][:3]}_{c['k']}_s{c['system']}"
                objectives[f"scan_fit_{sig}"] = "maximize"
                objectives[f"scan_extent_{sig}"] = "maximize"

        run_by_name = {e.name: e.run for e in exps}
        plan = _build_plan(goal, run_by_name, cycle)
        report_md = _report_for(cycle)
        print(f"goal: {goal[:120]}...")
        print(f"experiments: {[e.name for e in exps]}  layers: {plan.layers} "
              f"topo: {plan.topo_order}")
        print(f"objectives: {objectives}\n")

        # ── 书生[行动]: 全程驱动 run_research_program(观察→推理→行动闭环) ──
        out = run_research_program(
            goal=goal,
            experiments=exps,
            objectives_config=objectives,
            client=client, model=args.model, base_url=args.base_url,
            verify=grounding_verifier(),
            out_md=report_md,
            planner=lambda _g: plan,          # 需求拆解 -> DAG 分层
            layer_epochs=True,                # P-A: 分层流式结算
            replan_gate=True,                 # P-B: 层间重规划门
            early_stop_gate=True,             # P-C: 证据驱动提前终止门
            diagnostic_tools=_diagnostic_tools(),
            max_iterations=args.max_iters,
            min_iterations=min(args.min_iters, max(1, len(exps) - 1)),
            max_parallel=2,
            strictness=args.strictness,       # 判断层·分级护栏(默认0=零提示, 不强锁模型)
        )

        print("\n[program] explored=%d pruned=%d pareto_front=%d convergence=%s mutations=%d"
          % (out.explored, out.pruned, len(out.pareto_front), out.converred, out.mutations))
        print(f"[program] executed_cache={sorted(out.cache.keys())}")
        for b in out.pareto_front:
            print(f"  surv -> {b['name']}")
        print(f"[gate] {out.verdict} unsubstantiated={out.ungrounded} source={out.report_source}")
        print(f"[护栏] strictness={args.strictness} 判断层软提示 ×{len(out.judgment_hints or [])}")
        if out.consolidated:
            _c = out.consolidated
            print(f"[P-A layers_covered] {_c.get('epochs')}  stream_view_sections={len(_c.get('stream_view') or [])}")
            print(f"[P-B replan] checked={(_c.get('replan') or {}).get('checked')} "
                  f"skipped={len((_c.get('replan') or {}).get('skipped', []) or [])}")
            _es = _c.get('early_stop') or {}
            print(f"[P-C early_stop] verdict={_es.get('verdict')} stopped_after_layer={_es.get('stopped_after_layer')}")
        print(f"第{cycle}轮报告: {report_md}")
        last_report = out.report or (report_md.read_text(encoding="utf-8")
                                     if report_md.exists() else "")

    print("\n===== 长程任务结束: 连续完成 %d 轮(观察→推理→行动→报告→下一轮, 无人工停顿) ====="
          % args.cycles)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())