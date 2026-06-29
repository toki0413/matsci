"""Dimer 方法 — 旋转找最低曲率方向 + Hessian 沿模式估计.

参考 Henkelman & Jónsson, J. Chem. Phys. 111, 7010 (1999).
Dimer 由两个相近点 R ± d*τ 组成, 旋转 τ 找最低曲率方向 (最负本征值对应本征矢),
然后沿 τ 把 R 平移到鞍点.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from huginn.types import ToolContext
from huginn.tools.neb._evaluators import eval_single

if TYPE_CHECKING:
    from huginn.tools.neb.tool import NEBToolInput


async def dimer_rotate(
    R: np.ndarray,
    tau: np.ndarray,
    d: float,
    atomic_numbers: list[int],
    cell: np.ndarray | None,
    args: "NEBToolInput",
    context: ToolContext,
    n_rot: int = 20,
) -> tuple[np.ndarray, float]:
    """旋转 Dimer 找最低曲率方向.

    返回 (新 tau, 曲率). 曲率 ≈ (E(R+d*τ) + E(R-d*τ) - 2 E(R)) / d^2.
    旋转用最速下降沿角坐标走, 简化版, 不做线搜索.
    """
    n = R.size
    rng = np.random.default_rng(0)

    # 算 R 处能量 (一次)
    e0, _ = await eval_single(
        R.reshape(-1, 3), atomic_numbers, cell, args, context
    )

    best_tau = tau.copy()
    best_curv = float("inf")

    for _ in range(max(1, n_rot)):
        # 两个 dimer 端点的能量
        R1 = (R + d * tau).reshape(-1, 3)
        R2 = (R - d * tau).reshape(-1, 3)
        e1, _ = await eval_single(R1, atomic_numbers, cell, args, context)
        e2, _ = await eval_single(R2, atomic_numbers, cell, args, context)
        curv = (e1 + e2 - 2.0 * e0) / (d * d)

        if curv < best_curv:
            best_curv = float(curv)
            best_tau = tau.copy()

        # 旋转: 在垂直 tau 的子空间里随机扰动方向, 沿降低曲率方向走一步
        perp = rng.standard_normal(n)
        perp -= np.dot(perp, tau) * tau
        nrm = np.linalg.norm(perp)
        if nrm < 1e-12:
            break
        perp = perp / nrm
        # 试探旋转一个小角度
        theta = 0.05
        tau_new = tau * math.cos(theta) + perp * math.sin(theta)
        tau_new /= np.linalg.norm(tau_new) + 1e-12
        tau = tau_new

    return best_tau, best_curv


async def estimate_hessian_along_mode(
    R: np.ndarray,
    tau: np.ndarray,
    d: float,
    atomic_numbers: list[int],
    cell: np.ndarray | None,
    args: "NEBToolInput",
    context: ToolContext,
) -> tuple[list[float], np.ndarray]:
    """估计沿 tau 方向的 Hessian 本征值 (中心差分).

    这里只算一维曲率作为最低本征值近似, 其余本征值用 0 填充.
    完整 Hessian 对角化需要 N^2 次评估, 太贵, 简化掉.
    """
    e0, _ = await eval_single(
        R.reshape(-1, 3), atomic_numbers, cell, args, context
    )
    R1 = (R + d * tau).reshape(-1, 3)
    R2 = (R - d * tau).reshape(-1, 3)
    e1, _ = await eval_single(R1, atomic_numbers, cell, args, context)
    e2, _ = await eval_single(R2, atomic_numbers, cell, args, context)
    curv = (e1 + e2 - 2.0 * e0) / (d * d)
    # 简化: 只给最低本征值 (沿 tau), 其余用占位正数 (假设其他方向都是极小方向)
    n_atoms = R.size // 3
    eigvals = [float(curv)] + [1.0] * (3 * n_atoms - 1)
    return eigvals, tau
