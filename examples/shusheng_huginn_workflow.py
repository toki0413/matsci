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

# 收敛聚合头头数预算等通用治理参数走 run_research_program 默认, 本 demo 不再重复.

# ── 真实证据实验 (run 返回 {objectives, summary, success}) ────────────────
_RNG = np.random.default_rng(0)


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
    import numpy.polynomial.polynomial as PP
    N = 256
    mesh = np.linspace(-1, 1, N)
    P = Lg.legval(mesh, np.eye(basis)).T
    D2 = np.zeros((N, basis))
    for j in range(basis):
        d2 = PP.polyder(Lg.leg2poly(np.eye(j + 1)[:, j]), 2)
        acc = np.zeros_like(mesh)
        for kk, cc in enumerate(d2):
            acc += cc * mesh ** kk
        D2[:, j] = acc
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


def _make_experiments():
    """八个真实证据分支 → Experiment 列表(深化对照组 + 确认组)."""
    from huginn.research import Experiment

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


def _build_plan(goal: str, run_by_name: dict):
    """需求拆解 -> 带依赖 DAG -> 分层.

    层0: X1 (判据基准, 无依赖)
    层1: X4/N平台 + X5/约束曲线 + X6/预算扫描 (对照, 依赖判据校准)
    层2: X7/多系统 (依赖约束曲线) + X8/k1位置依赖 (依赖约束曲线) + X2/X3 (确认组)
    """
    from huginn.research.planning import build_research_plan, SubResearch
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry", action="store_true", help="确定性运行(不调模型), 验证整条管线")
    ap.add_argument("--model", default="intern-s2-preview")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--max-iters", type=int, default=12)
    ap.add_argument("--min-iters", type=int, default=3)
    ap.add_argument("--strictness", type=int, default=0, choices=[0, 1, 2],
                    help="判断层·分级护栏 0/1/2(默认0=纯真伪硬门禁, 信任模型)")
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

    exps = _make_experiments()
    run_by_name = {e.name: e.run for e in exps}
    plan = _build_plan(GOAL, run_by_name)

    print("== 书生 + Huginn 完整深研管线 ==")
    print(f"goals: {GOAL}")
    print(f"layers: {plan.layers}  topo: {plan.topo_order}  antichain_width={plan.antichain_width}")
    print(f"objectives: {_OBJECTIVES}\n")

    out = run_research_program(
        goal=GOAL,
        experiments=exps,
        objectives_config=_OBJECTIVES,
        client=client, model=args.model, base_url=args.base_url,
        verify=grounding_verifier(),
        out_md=_OUT / "research_report.md",
        planner=lambda _g: plan,          # 需求拆解 -> DAG 分层
        layer_epochs=True,                # P-A: 分层流式结算
        replan_gate=True,                 # P-B: 层间重规划门
        early_stop_gate=True,             # P-C: 证据驱动提前终止门
        diagnostic_tools=_diagnostic_tools(),
        max_iterations=args.max_iters,
        min_iterations=args.min_iters,
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
    print("报告:", _OUT / "research_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())