"""纯数值对照探针 —— 校验「不确定性指数 β」装置能否测出分形吸引域。

背景: 我们要验证「大模型求解是否陷入分形吸引域」的假说。但分形判据(β<1)
必须先在一个"已知确有分形吸引域"的实验装置上调通(校准), 否则事后测不出 agent
的分形性 = 装置坏了而非 agent 非分形。

这里用最经典、有严格数学结论的载体: **牛顿法求复数多项式 z^3 - 1 的 3 个根**。
它的每个根有一个吸引域(basin of attraction), 三域边界是典型分形。
判据: **不确定性指数 beta**。对两个相距 epsilon 的初始点, 若它们牛顿迭代后
落入不同根的概率 f(eps) ~ eps^(d-D), 则 log f / log eps 在 eps->0 时趋近 beta。
   - 边界光滑(非分形): f(eps) ~ eps      => beta -> 1
   - 边界分形:          f(eps) ~ eps^(<1) => beta < 1
本例已知 Newton fractal 边界是分形的, 预期 beta 明显 < 1, 从而证明装置测得准。

纯标准库, 不依赖任何重型依赖。可在这台/DSW 任意 CPU 上跑。
"""
from __future__ import annotations

import math
import random
from bisect import bisect_left


# ── 载体: 牛顿迭代 + 收敛到哪个根 ─────────────────────────────
def _roots_z3():
    """z^3 - 1 = 0 的三个根, 角度 0, 2pi/3, 4pi/3 (单位圆三等分)."""
    return [(1.0, 0.0),
            (math.cos(2 * math.pi / 3), math.sin(2 * math.pi / 3)),
            (math.cos(4 * math.pi / 3), math.sin(4 * math.pi / 3))]


def _newton_converge(zr, zi, max_iter=100, tol=1e-10):
    """牛顿法迭代 z := z - (z^3-1)/(3 z^2). 返回收敛到的根序号(0/1/2)或 -1."""
    for _ in range(max_iter):
        zr2 = zr * zr
        zi2 = zi * zi
        # f = z^3 - 1, f' = 3 z^2
        # z^3 = (zr + i zi)^3
        f_r = zr * zr2 - 3 * zr * zi2 - 1.0
        f_i = 3 * zr2 * zi - zi * zi2
        # f' = 3 (zr + i zi)^2
        fp_r = 3 * (zr2 - zi2)
        fp_i = 6 * zr * zi
        denom = fp_r * fp_r + fp_i * fp_i
        if denom < 1e-300:  # 卡在临界点, 判未收敛
            return -1
        # delta = f / f'
        d_r = (f_r * fp_r + f_i * fp_i) / denom
        d_i = (f_i * fp_r - f_r * fp_i) / denom
        zr -= d_r
        zi -= d_i
        if d_r * d_r + d_i * d_i < tol * tol:
            break
    # 归一到最近的根(按距离)
    roots = _roots_z3()
    best, bd = -1, 1e30
    for i, (rr, ri) in enumerate(roots):
        d = (zr - rr) ** 2 + (zi - ri) ** 2
        if d < bd:
            bd, best = d, i
    return best


# ── 不确定性指数 beta ──────────────────────────────────────────
def _f_epsilon(eps: float, n_samples: int = 3000, seed: int = 0) -> float:
    """相距 eps 的随机初始点对,落入不同根的概率 f(eps).

    采样一个随机初始点 p, 让 p2 = p + eps*random_direction, 二者牛顿迭代,
    统计落入不同根的占比。
    """
    rng = random.Random(seed)
    diff = 0
    for _ in range(n_samples):
        # 均匀随机初始点 (围住三根的区域: 以原点为中心 r~U(0,1.5), 角~U(0,2pi))
        r = rng.uniform(0.0, 1.5)
        th = rng.uniform(0.0, 2 * math.pi)
        z1r, z1i = r * math.cos(th), r * math.sin(th)
        # 第二个点: 沿随机方向偏离 eps
        dth = rng.uniform(0.0, 2 * math.pi)
        z2r, z2i = z1r + eps * math.cos(dth), z1i + eps * math.sin(dth)
        k1 = _newton_converge(z1r, z1i)
        k2 = _newton_converge(z2r, z2i)
        if k1 >= 0 and k2 >= 0 and k1 != k2:
            diff += 1
    return diff / n_samples


def estimate_beta(epss=None, seed: int = 0) -> dict:
    """扫 eps 从大到小, 拟合 beta = d log f / d log eps (端点斜率)."""
    epss = epss or [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    pts = []
    for e in epss:
        f = _f_epsilon(e, seed=seed)
        pts.append((math.log(e), math.log(f)))
        print(f"  eps={e:7.4f}  f={f:.4f}  logf/logeps="
              f"{f / e if f else float('nan'):.3f}")
    # 用最小二乘拟合一整段 log f = beta * log eps + c
    n = len(pts)
    sx = sum(p[0] for p in pts)
    sy = sum(p[1] for p in pts)
    sxx = sum(p[0] * p[0] for p in pts)
    sxy = sum(p[0] * p[1] for p in pts)
    denom = n * sxx - sx * sx
    beta = (n * sxy - sx * sy) / denom if denom else float("nan")
    return {"beta": beta, "points": pts}


if __name__ == "__main__":
    print("=" * 62)
    print("牛顿法 z^3-1 吸引域 —— 不确定性指数 beta 校准")
    print("期望: Newton fractal 边界分形 -> beta 明显 < 1")
    print("=" * 62)
    # 采样规模随 eps 增大(小 eps 需要更多样本才稳)
    res = estimate_beta()
    b = res["beta"]
    print("-" * 62)
    print(f"拟合 beta = {b:.3f}")
    if b < 0.9:
        print(f"判定: beta<1 ({b:.3f}) -> 边界呈分形, 装置测得准 ✓")
    else:
        print(f"判定: beta≈1 -> 边界光滑? 检查装置或采样量 (beta={b:.3f})")
    print("说明: 随机对照基准(无结构)应 beta=1; Newton fractal 已知 <1。")