"""research_imagination_basins —— 动机B: 想象末态语义盆地边界的分形性探针.

背景(承接分形假说双路负结论): 我们拒绝"为凑 β<1 而给 agent 引入折叠"(动机A,
目标倒置)。但若想象力在主循环里**本就构成反复的"想象→扰动→再想象"链**(动机B),
那它就在预测/假设结果分布上构成真实轨迹, 值得测一次末态盆地边界维数。

零 LLM(与 research_a / probe_failure_basins 同风格): 用确定性、可复现的
"想象动力学代理"替代真实 LLM 变换 —— 明确它是代理, 不冒充真实大脑而言:
  结构变换三族 = imagination.py 的 algebraic/topological/order,
  映射自结构规则推导, 而非硬塞 Rössler/Newton 混沌.

探针模型(三段机制对标"分形盆地所需"的动力学):
  1. 连续参数空间: 状态 z∈R²(两条可观测预测/自洽不变量), 初值在[-1,1]² 连续变化
  2. 多个自洽"吸引子机制": 三族的稳态(yi i 处 as 局部稳定), 终态归属 argmin|z-c_i|
  3. 非线性迭代 + 折叠: sector(atan2) 决定下一步用哪一族正则映射; 有界状态空间
     越界处以取余折叠(fold)拉回 ——"拉伸(section of λ)+有界域折叠"是物理合理的,
     非硬造分形. 拉越强(λ→大)折叠越频繁, 边界可能由光滑变分形.

  让数据决定: 扫拉伸强度 λ, 得 β(λ) 曲线. 若某 λ 下细/粗δ斜率一致且稳定<1 → 分形;
  若细→1 粗→0 → 有限直边(非分形). 不动手为测分形而挑 λ(诚实边界).

测量协议与 research_a 完全一致:
  P_cross(δ): grid 上相距 δ 两点落入不同末态被测域比例 → log P ~ β log δ
  光滑对照(竖直线→β≈1) / 噪声对照(随机→β≈0) 校验估计器.
"""
from __future__ import annotations

import math

import numpy as np

# ── 除 JSON 序除常量 ─────────────────────────────────────────────
DELTAS = [0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.25, 0.33]
D_FINE = [0.02, 0.03, 0.045, 0.06]
D_COARSE = [0.12, 0.16, 0.22, 0.33]
_STEPS = 60  # 想象链迭代步数

# 三族稳态(暂标: 一阶海 tesla triangular vertices) — 三族机制的自洽目标
_CENTERS = np.asarray([[1.0, 0.0], [-0.5, 0.8660254], [-0.5, -0.8660254]])


def _fold(x, L=1.0):
    """有界域折叠: 把 x 用取余拉回 [-L, L], 越界即"折叠一次"."""
    return ((x + L) % (2 * L)) - L


def iterate_labels(a0: np.ndarray, b0: np.ndarray, lam: float,
                   theta: float, steps: int):
    """从 (a0,b0) 网格迭代想象链 → 每点终态归属标签(0/1/2).

    向量化整网格一次广播全部点 N 步. 每一步:
      sector = int(3 * (atan2(b,a)+π) / 2π) % 3  → 决定本步用哪一族
      alg:    旋转(对称守恒)
      topo:   (λa, λb) → 候选拉伸 → 越界处折叠(关键折叠源)
      order:  反射/镜像
    每步后统一 fold 回有界域. 末态 argmin 到三稳态 → 被测域标签.
    返回 (label_grid, final_state_hash简介).
    """
    z = np.stack([a0.astype(np.float64), b0.astype(np.float64)], axis=-1)
    for _ in range(steps):
        ang = np.arctan2(z[..., 1], z[..., 0])
        sector = (np.floor(3.0 * (ang + np.pi) / (2.0 * np.pi))).astype(int) % 3
        ct, st = math.cos(theta), math.sin(theta)
        # 三族统一算出, 按 sector 选
        # alg: 旋转
        z_alg = np.stack([ct * z[..., 0] - st * z[..., 1],
                          st * z[..., 0] + ct * z[..., 1]], axis=-1)
        # topo: 拉伸
        z_topo = np.stack([lam * z[..., 0], lam * z[..., 1]], axis=-1)
        # order: 反射
        z_order = np.stack([z[..., 0], -z[..., 1]], axis=-1)
        z = np.where(sector[..., None] == 0, z_alg,
                     np.where(sector[..., None] == 1, z_topo, z_order))
        z = _fold(z)  # 统一折叠回有界域
    d = np.linalg.norm(z[..., None, :] - _CENTERS[None, None, :, :], axis=-1)  # (g,g,3)
    labels = np.argmin(d, axis=-1)
    return labels


# ── 测量协议(与 research_a 一致) ────────────────────────────────
def _u01(n):
    return [i / (n - 1) if n > 1 else 0.5 for i in range(n)]


def _label_grid(lam, theta, grid_n, steps=_STEPS):
    ns = np.asarray(_u01(grid_n))
    a0, b0 = 2 * ns - 1, 2 * ns - 1
    A, B = a0[:, None], b0[None, :]
    return iterate_labels(np.tile(A, (1, grid_n)), np.tile(B, (grid_n, 1)),
                          lam, theta, steps)


def _p_cross(g: np.ndarray, grid_n: int, delta_frac: float, rng, samples: int = 8000) -> float:
    d = max(1, int(round(delta_frac * (grid_n - 1))))
    steps_vec = np.asarray([(-d, 0), (d, 0), (0, -d), (0, d),
                            (d, d), (-d, d), (d, -d), (-d, -d)], dtype=int)
    i = rng.integers(0, grid_n, size=samples)
    j = rng.integers(0, grid_n, size=samples)
    s = rng.integers(0, len(steps_vec), size=samples)
    i2, j2 = i + steps_vec[s, 0], j + steps_vec[s, 1]
    ok = (i2 >= 0) & (i2 < grid_n) & (j2 >= 0) & (j2 < grid_n)
    if not ok.any():
        return 0.0
    return float(1.0 - (g[i[ok], j[ok]] == g[i2[ok], j2[ok]]).mean())


def _fit_beta_r2(deltas, pcs) -> dict:
    pts = [(math.log(x), math.log(max(p, 1e-9))) for x, p in zip(deltas, pcs)
           if 0.0 < p < 1.0]
    if len(pts) < 3:
        return {"beta": float("nan"), "r2": float("nan"), "n_used": len(pts)}
    xs = [t[0] for t in pts]
    ys = [t[1] for t in pts]
    slope, intercept = np.polyfit(xs, ys, 1)
    yhat = [slope * x + intercept for x in xs]
    ss_res = sum((y - yh) ** 2 for y, yh in zip(ys, yhat))
    ss_tot = sum((y - sum(ys) / len(ys)) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot else float("nan")
    return {"beta": float(slope), "r2": float(r2), "n_used": len(pts)}


def estimate_basin_beta(lam, theta, grid_n=160, seed=3, steps=_STEPS) -> dict:
    """单 λ: 末态盆地 β + 细/粗δ局部斜率 + 光滑/噪声对照."""
    g = _label_grid(lam, theta, grid_n, steps)
    rng = np.random.default_rng(seed)
    pcs = [_p_cross(g, grid_n, d, rng) for d in DELTAS]
    fit = _fit_beta_r2(DELTAS, pcs)
    fine = _fit_beta_r2(D_FINE, [_p_cross(g, grid_n, d, rng) for d in D_FINE])["beta"]
    coarse = _fit_beta_r2(D_COARSE, [_p_cross(g, grid_n, d, rng) for d in D_COARSE])["beta"]
    # 对照: 光滑(竖直线) / 噪声(随机)
    g_smooth = np.asarray([[0 if u < 0.5 else 1 for _ in _u01(grid_n)]
                           for u in _u01(grid_n)], dtype=int)
    rng_s = np.random.default_rng(seed)
    bet_s = _fit_beta_r2(DELTAS, [_p_cross(g_smooth, grid_n, d, rng_s) for d in DELTAS])["beta"]
    rng_n = np.random.default_rng(seed)
    g_noise = np.asarray([[rng_s.integers(0, 3) for _ in _u01(grid_n)] for _ in _u01(grid_n)],
                         dtype=int)
    bet_n = _fit_beta_r2(DELTAS, [_p_cross(g_noise, grid_n, d, rng_n) for d in DELTAS])["beta"]
    return {"lam": lam, "theta": theta, "overall": fit, "fine": fine, "coarse": coarse,
            "smooth_ctl": bet_s, "noise_ctl": bet_n}


if __name__ == "__main__":
    print("=" * 70)
    print("动机B探针: 想象末态盆地边界分形性  (零LLM 数值代理, 扫描拉伸强度 λ)")
    print("β(λ) 曲线; 细δ→1/粗δ→0=有限直边(非分形); 细粗一致且稳定<1=分形")
    print("=" * 70)
    theta = 2.4  # 对称旋转角(略小于π, 固定的结构常数)
    lams = [1.0, 1.5, 2.0, 2.6, 3.2]
    print(f"{'λ':>5} {'总体β':>7} {'R²':>6} {'细δβ':>7} {'粗δβ':>7}  "
          f"{'光滑对照':>7} {'噪声对照':>7}  判读")
    for lam in lams:
        r = estimate_basin_beta(lam, theta)
        b, r2 = r["overall"]["beta"], r["overall"]["r2"]
        fine, coarse = r["fine"], r["coarse"]
        noise = r["noise_ctl"]
        spread = coarse - fine
        if math.isnan(b):
            verdict = "无边界(末态恒一)"
        elif b < 0.12 or b < noise + 0.08:
            # β 掉进噪声对照同量级带 → 边界纠缠到与随机打标不可分(混沌极限),
            # 不是"优雅分形". 诚实边界: 与噪声不可分, 不叫分形.
            verdict = "混沌纠缠(≈噪声, 与随机不可分)"
        elif spread > 0.35 or fine > 0.9:
            verdict = "有限直边, 非分形"
        elif abs(spread) < 0.25 and b < 0.95:
            verdict = "真分形(细粗斜率一致且稳定<1)"
        else:
            verdict = "介乎, 需更宽δ窗"
        print(f"{lam:>5} {b:>+7.3f} {r2:>6.3f} "
              f"{fine:>+7.3f} {coarse:>+7.3f}  "
              f"{r['smooth_ctl']:>+7.3f} {noise:>+7.3f}  {verdict}")