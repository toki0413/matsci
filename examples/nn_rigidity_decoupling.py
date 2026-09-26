"""架构容量外推解耦实验 —— N_c(w) 是解的性质还是模型的性质?

刚性问题锚点 (非齐次 ODE, 无平凡解陷阱):
    u''(x) = f(x)  on [0,1]
    poly: u*(x) = x^2 + 0.3x - 0.2   -> 解空间 { x^2 + ax + b }, 2 维自由 (a,b)
    加 N 个点值约束 -> 当 N>=2 且点一般 -> a,b 唯一 -> 解空间塌成 0 维 (刚性)

为什么不用 u''+u=0:
    该齐次方程存在平凡解 u≡0, PINN 会坍缩到 u≈0 的真实局部极小 (实测 loss 恒为
    0.636, 与宽度/学习率/种子无关). 换成非齐次 (f≠0) 后平凡解不再是解, 优化可跑通.
    这本身就是"优化可达性"混淆 (报告 §1c) 的一个实证.

测什么:
    固定容量 w, 扫描约束数 N, 令
        N_c(w) = min{ N : 对所有 N'>=N 都有 V_ho(w,N') < VHO_CUT }
    即"留出违规首次跌破阈值且此后不再回升"的拐点. 再由 N_c(w) ~ w^beta 判定:
        beta ~ 0    -> 饱和点与容量无关 -> 是解的性质   -> 探针成立 (H1)
        beta > 0.3  -> 饱和点随容量增长 -> 是模型记忆容量 -> 归纳偏置假象 (H2)

为何能解耦:
    解空间自由度固定, 不随网络加宽而变; 模型记忆容量却随 w 增长.
    原研究只用同一个 V_ho 既诊断优化又诊断维度 -> 信号共线. 这里沿 w 外推,
    把"与 C 无关的解属性"与"与 C 有关的模型属性"分开.

阈值语义 (重要):
    V_tr / V_ho 均取**均方**违规 (mean of squared violation), 不是 RMS.
        V_tr = mean_{训练点}(残差^2 + 点值^2)          <- 门槛: < VTR_GATE
        V_ho = mean_{留出点}(残差^2 + 点值^2)          <- N_c 判据: < VHO_CUT
    报告原文写 "1e-10 / 1e-8", 但未说明是均方还是 RMS; 由于 sqrt(1e-20)=1e-10,
    按 RMS 解释要求均方 <1e-20, 实测不可达. 故统一按均方口径, 并保留 VTR_GATE(1e-10)
    这一档以匹配原文数量级: 收敛模型实测 V_tr~1e-13, 未收敛~1e-3, 分离度 10 个量级.

优化管道 (前置门槛):
    阶段1 Adam(5000, lr=1e-3) -> 阶段2 scipy L-BFGS-B (解析梯度, 可多次 warm-restart).
    torch 自带 LBFGS 实测卡在 loss~1e-6 的平台 (梯度 ~1e-6 无法下降);
    scipy L-BFGS-B 可到 1e-12~1e-16, 这是本实验能成立的前提.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize

torch.set_default_dtype(torch.float64)

_PI = math.pi
_OUT = Path(__file__).resolve().parent / "out" / "nn_rigidity_decoupling"

# ── 判据阈值 (均方口径) ─────────────────────────────────────────────────────
VTR_GATE = 1e-10   # 优化充分性门槛: 训练均方违规须低于此值, 否则该种子作废
VHO_CUT = 1e-8     # 留出均方违规"零违规"阈值 (报告 §D 原值), 用于定义 N_c
ADAM_STEPS = 5000
ADAM_LR = 1e-3


# ── 问题锚点 ───────────────────────────────────────────────────────────────
def _anchor(kind: str):
    if kind == "poly":
        # u* = x^2 + 0.3x - 0.2,  u*'' = 2  -> 解空间 {x^2 + ax + b}, 2 维
        u_star = lambda t: t * t + 0.3 * t - 0.2
        f = lambda t: torch.full_like(t, 2.0)
        desc = "u''=2, u*=x^2+0.3x-0.2, 解空间 {x^2+ax+b} (2 DOF)"
    elif kind == "osc":
        # u* = cos(2pi x) + 0.3x - 0.2, u*'' = -(2pi)^2 cos(2pi x)
        u_star = lambda t: torch.cos(2 * _PI * t) + 0.3 * t - 0.2
        f = lambda t: -(2 * _PI) ** 2 * torch.cos(2 * _PI * t)
        desc = "u''=-(2pi)^2 cos(2pi x), u*=cos(2pi x)+0.3x-0.2 (2 DOF)"
    elif kind == "hi":
        # 高频但**仍然刚性**: 解空间仍为 {x 一次支 + 特定非齐次特解}, 2 DOF.
        # 唯一性与 poly/osc 完全相同; 差别只在"小网络能否表达 cos(8pi x)".
        # 这是报告 §4(1) "假刚性/假胖" (架构表达瓶颈) 的直接对照臂:
        # 若某个 w 的 N_c 因表达瓶颈而抬高, 就会出现 beta>0, 而真问题并未变胖.
        u_star = lambda t: torch.cos(8 * _PI * t) + 0.3 * t - 0.2
        f = lambda t: -(8 * _PI) ** 2 * torch.cos(8 * _PI * t)
        desc = "u''=-(8pi)^2 cos(8pi x), u*=cos(8pi x)+0.3x-0.2 (2 DOF, 高频表达瓶颈对照)"
    elif kind == "hism":
        # 与 hi 同频 (8pi) 但**把 forcing 归一到 O(1)**: u*'' = -cos(8pi x).
        # 解空间仍为 {cos(8pi x)/(8pi)^2 + ax + b}, 2 DOF, 与 hi 完全同构.
        # 唯一差别是残差量纲: hi 的 |f|~632 -> V_tr 天然 ~4e5 (数值尺度混淆),
        # hism 的 |f|~1 -> V_tr O(1). 用它判定 hi 的失败究竟来自
        #   (a) 数值尺度 (hism 会收敛) 还是 (b) 高频表达瓶颈/优化条件 (hism 也失败).
        C = (8 * _PI) ** 2
        u_star = lambda t: torch.cos(8 * _PI * t) / C + 0.3 * t - 0.2
        f = lambda t: -torch.cos(8 * _PI * t)
        desc = "u''=-cos(8pi x), u*=cos(8pi x)/(8pi)^2+0.3x-0.2 (2 DOF, 频率同 hi 但 forcing 归一)"
    elif kind == "fat":
        # **真胖对照臂**: PDE 仍为 u''=2 (解空间 {x^2+ax+b}, 2 DOF), 但 N 个点值约束
        # 全部落在同一点 x=0.5 -> 约束秩恒为 1, **不随 N 增长** -> 'a' 永远自由,
        # V_ho = a^2 * mean((x-0.5)^2) 恒为 O(1), 对任何 N / 任何 w 都不饱和.
        # 这是"解空间确实胖(连续族未被约束切掉)"的极简解析实现, 与 poly 只差约束点位置.
        # 用它检验: 判据在**真胖**时是否真的给出 N_c=None (阴性对照 / 探针灵敏度).
        u_star = lambda t: t * t + 0.3 * t - 0.2
        f = lambda t: torch.full_like(t, 2.0)
        desc = "u''=2, u*=x^2+0.3x-0.2; N 个点值约束全在 x=0.5 (秩恒1, 'a'恒自由) -> 真胖"
    else:
        raise ValueError(kind)
    return u_star, f, desc


# ── 模型 ───────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    """3 隐层 Tanh, 宽度 w (与协议/前序实测一致)."""

    def __init__(self, w: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, w), nn.Tanh(),
            nn.Linear(w, w), nn.Tanh(),
            nn.Linear(w, w), nn.Tanh(),
            nn.Linear(w, 1),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t).squeeze(-1)


# ── 点集 ───────────────────────────────────────────────────────────────────
def _colloc(m: int) -> torch.Tensor:
    return torch.linspace(0.0, 1.0, m).reshape(-1, 1)


def _train_pts(n: int, kind: str = "poly") -> torch.Tensor:
    """训练点值约束的位置.

    默认均匀铺在 [0,1]; ``fat`` 锚点把它们**全部堆在 x=0.5**, 使约束秩恒为 1,
    从而让解族的自由参数 'a' 不随 N 被切掉 (真胖对照).
    """
    if kind == "fat":
        return torch.full((n, 1), 0.5)
    return torch.linspace(0.0, 1.0, n).reshape(-1, 1)


def _holdout(k: int = 128) -> torch.Tensor:
    """留出点: 128 格的中点, 与 linspace(N)/colloc 一般不重合."""
    return (torch.linspace(0.0, 1.0, k + 1)[:-1] + 0.5 / k).reshape(-1, 1)


# ── 残差 ───────────────────────────────────────────────────────────────────
def _d2(net: MLP, t: torch.Tensor, create_graph: bool) -> torch.Tensor:
    tt = t.clone().requires_grad_(True)
    U = net(tt)
    dU = torch.autograd.grad(U, tt, torch.ones_like(U), create_graph=True)[0]
    d2U = torch.autograd.grad(dU, tt, torch.ones_like(dU), create_graph=create_graph)[0]
    return d2U


# ── 单次求解 ───────────────────────────────────────────────────────────────
def solve(kind: str, w: int, n: int, seed: int, *,
          m_colloc: int = 64, adam_steps: int = ADAM_STEPS,
          adam_lr: float = ADAM_LR, lbfgs_rounds: int = 3,
          lbfgs_maxiter: int = 4000) -> dict:
    torch.set_num_threads(1)
    u_star, f, _ = _anchor(kind)
    torch.manual_seed(seed)
    net = MLP(w)

    tc = _colloc(m_colloc)
    tp = _train_pts(n, kind)
    yp = u_star(tp.squeeze(-1))
    th = _holdout()
    yh = u_star(th.squeeze(-1))
    fc = f(tc)
    fh = f(th)

    def res_tr() -> torch.Tensor:
        return _d2(net, tc, True) - fc

    def loss() -> torch.Tensor:
        return res_tr().pow(2).mean() + (net(tp) - yp).pow(2).mean()

    # 阶段 1: Adam
    opt = torch.optim.Adam(net.parameters(), lr=adam_lr)
    for _ in range(adam_steps):
        opt.zero_grad()
        L = loss()
        L.backward()
        opt.step()

    # 阶段 2: scipy L-BFGS-B (解析梯度), 可 warm-restart
    def fg(x: np.ndarray):
        with torch.no_grad():
            i = 0
            for p in net.parameters():
                k = p.numel()
                p.copy_(torch.tensor(x[i:i + k], dtype=torch.float64).reshape(p.shape))
                i += k
        net.zero_grad()
        L = loss()
        L.backward()
        g = np.concatenate([p.grad.detach().numpy().ravel() for p in net.parameters()])
        return float(L.detach()), g

    x = np.concatenate([p.detach().numpy().ravel() for p in net.parameters()])
    converged = False
    for _ in range(lbfgs_rounds):
        r = minimize(fg, x, jac=True, method="L-BFGS-B",
                     options={"maxiter": lbfgs_maxiter, "ftol": 1e-22,
                              "gtol": 1e-15, "maxcor": 50})
        x = r.x
        converged = float(np.abs(r.jac).max()) < 1e-9
        if converged:
            break

    # 观测量 (均方口径)
    with torch.no_grad():
        m = 0
        for p in net.parameters():
            k = p.numel()
            p.copy_(torch.tensor(x[m:m + k], dtype=torch.float64).reshape(p.shape))
            m += k

    res_tr_v = (_d2(net, tc, False) - fc).detach()
    v_tr_v = (net(tp) - yp).detach()
    res_ho_v = (_d2(net, th, False) - fh).detach()
    v_ho_v = (net(th) - yh).detach()

    v_tr = float(res_tr_v.pow(2).mean() + v_tr_v.pow(2).mean())
    v_ho = float(res_ho_v.pow(2).mean() + v_ho_v.pow(2).mean())
    return {
        "kind": kind, "w": w, "n": n, "seed": seed,
        "v_tr": v_tr, "v_ho": v_ho,
        "res_ho_rms": float(res_ho_v.pow(2).mean().sqrt()),
        "val_ho_rms": float(v_ho_v.pow(2).mean().sqrt()),
        "converged": converged,
    }


# ── 判读 ───────────────────────────────────────────────────────────────────
def _median(xs):
    s = sorted(xs)
    k = len(s)
    if k == 0:
        return float("nan")
    return s[k // 2] if k % 2 else 0.5 * (s[k // 2 - 1] + s[k // 2])


def compute_nc(ns, vho_med, cut=VHO_CUT):
    """N_c = min{ N : 对所有 N'>=N 都有 V_ho < cut }. 无解返回 None."""
    for i in range(len(ns)):
        if all(v < cut for v in vho_med[i:]):
            return ns[i]
    return None


def fit_beta(ws, ncs):
    pts = [(w, nc) for w, nc in zip(ws, ncs) if nc is not None]
    if len(pts) < 2:
        return None, None, len(pts)
    xs = [math.log(w) for w, _ in pts]
    ys = [math.log(nc) for _, nc in pts]
    k = len(xs)
    mx, my = sum(xs) / k, sum(ys) / k
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx <= 0:
        return 0.0, 1.0, k          # 所有 N_c 相同 -> 完全水平
    beta = sxy / sxx
    a = my - beta * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a + beta * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return beta, r2, k


# ── 主流程 ─────────────────────────────────────────────────────────────────
def _task(args):
    return solve(**args)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="poly", choices=["poly", "osc", "hi", "hism", "fat"])
    ap.add_argument("--widths", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    ap.add_argument("--ns", type=int, nargs="+", default=[2, 3, 4, 6, 8, 16, 32, 64])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--colloc", type=int, default=64)
    ap.add_argument("--adam-steps", type=int, default=ADAM_STEPS)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    _, _, desc = _anchor(args.kind)
    print(f"锚点: {desc}")
    print(f"网格: w={args.widths}  N={args.ns}  seeds={args.seeds}  "
          f"colloc={args.colloc}  adam={args.adam_steps}  jobs={args.jobs}")
    print(f"门槛: V_tr < {VTR_GATE:.0e} (均方)  |  N_c 判据: V_ho < {VHO_CUT:.0e} (均方)")
    print("=" * 104, flush=True)

    tasks = [dict(kind=args.kind, w=w, n=n, seed=s, m_colloc=args.colloc,
                  adam_steps=args.adam_steps)
             for w in args.widths for n in args.ns for s in range(args.seeds)]

    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for i, r in enumerate(ex.map(_task, tasks, chunksize=1), 1):
            results.append(r)
            if i % 10 == 0 or i == len(tasks):
                el = time.time() - t0
                print(f"  [{i}/{len(tasks)}] {el:6.0f}s  (ETA {el / i * (len(tasks) - i):5.0f}s)",
                      flush=True)

    # 汇总
    rows = []
    for w in args.widths:
        for n in args.ns:
            vs = [r for r in results if r["w"] == w and r["n"] == n]
            good = [r for r in vs if r["v_tr"] < VTR_GATE]
            rows.append({
                "w": w, "n": n,
                "pass_frac": len(good) / len(vs),
                "v_tr_med": _median([r["v_tr"] for r in vs]),
                "v_ho_med_all": _median([r["v_ho"] for r in vs]),
                "v_ho_med_good": _median([r["v_ho"] for r in good]) if good else float("nan"),
                "res_ho_rms_good": _median([r["res_ho_rms"] for r in good]) if good else float("nan"),
                "val_ho_rms_good": _median([r["val_ho_rms"] for r in good]) if good else float("nan"),
            })

    print("=" * 104)
    print(f"{'w':>5} {'N':>4} | {'pass':>5} | {'V_tr(med)':>10} | "
          f"{'V_ho(med)':>10} | {'V_ho(good)':>10} | {'res_ho_rms':>10} {'val_ho_rms':>10}")
    print("-" * 104)
    for r in rows:
        print(f"{r['w']:>5} {r['n']:>4} | {r['pass_frac']:>5.0%} | {r['v_tr_med']:>10.2e} | "
              f"{r['v_ho_med_all']:>10.2e} | {r['v_ho_med_good']:>10.2e} | "
              f"{r['res_ho_rms_good']:>10.2e} {r['val_ho_rms_good']:>10.2e}")

    # N_c(w) 与 beta
    print("=" * 104)
    ncs, ncs_good = [], []
    for w in args.widths:
        a = [next(r["v_ho_med_all"] for r in rows if r["w"] == w and r["n"] == n) for n in args.ns]
        b = [next(r["v_ho_med_good"] for r in rows if r["w"] == w and r["n"] == n) for n in args.ns]
        nc_a, nc_b = compute_nc(args.ns, a), compute_nc(args.ns, b)
        ncs.append(nc_a)
        ncs_good.append(nc_b)
        print(f"w={w:>4} | N_c(全部种子)={nc_a}   N_c(仅合规种子)={nc_b}")

    print("-" * 104)
    fits = {}
    for label, ncs_ in (("全部种子", ncs), ("仅合规种子", ncs_good)):
        beta, r2, k = fit_beta(args.widths, ncs_)
        fits[label] = {"beta": beta, "r2": r2, "k": k}
        if beta is None:
            print(f"[{label}] 有限 N_c 点不足 ({k}), 无法拟合 beta")
        else:
            print(f"[{label}] N_c(w) ~ w^beta :  beta={beta:+.3f}  R²={r2:.3f}  (用 {k}/{len(args.widths)} 宽度)")

    beta = fits["全部种子"]["beta"]
    r2 = fits["全部种子"]["r2"]
    if beta is None or not math.isfinite(beta):
        verdict = "无法判定: 有效 N_c 点不足 (大量配置无饱和)"
    elif abs(beta) < 0.15:
        verdict = "H1: 饱和点与容量无关 -> N_c 是解的性质, 探针站得住"
    elif beta > 0.3 and (r2 or 0) > 0.85:
        verdict = "H2: 饱和点随容量增长 -> 归纳偏置假象, 探针被驳回"
    else:
        verdict = "灰区: beta 介于两者之间, 需加大网格/种子数再判"
    print(f"\n裁决: {verdict}")

    _OUT.mkdir(parents=True, exist_ok=True)
    out = _OUT / f"decoupling_{args.kind}{('_' + args.tag) if args.tag else ''}.json"
    out.write_text(json.dumps({
        "config": vars(args), "anchor": desc,
        "thresholds": {"vtr_gate": VTR_GATE, "vho_cut": VHO_CUT, "basis": "mean_squared"},
        "rows": rows,
        "N_c": {"all": dict(zip(map(str, args.widths), ncs)),
                "good": dict(zip(map(str, args.widths), ncs_good))},
        "fit": fits, "verdict": verdict, "elapsed_s": time.time() - t0,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果 -> {out}  (耗时 {time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())