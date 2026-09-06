"""NEB 核心算法 — IDPP 初猜 / 力投影 / 势垒 / 路径长度.

参考:
  - IDPP: Smidstrup et al., J. Chem. Phys. 140, 214106 (2014)
  - NEB 力投影 (improved tangent): Henkelman & Jónsson, J. Chem. Phys. 113, 9901 (2000)
"""
from __future__ import annotations

import numpy as np


def idpp_initial_path(
    init_pos: np.ndarray, final_pos: np.ndarray, n_images: int
) -> list[np.ndarray]:
    """IDPP (Image Dependent Pair Potential) 初猜.

    先线性插值得到一组路径, 再对每个中间 image 优化其原子位置,
    使得该 image 的 pairwise 距离矩阵匹配 D_target(t) = (1-t)*D_init + t*D_final.
    目标函数: sum_{j<k} (|r_j-r_k| - D_target[j,k])^2 / D_target[j,k]^2.
    """
    # 线性插值
    ts = np.linspace(0.0, 1.0, n_images)
    images = [
        (1.0 - t) * init_pos + t * final_pos for t in ts
    ]

    # 距离矩阵
    def pairwise_dists(pos: np.ndarray) -> np.ndarray:
        diff = pos[:, None, :] - pos[None, :, :]
        return np.linalg.norm(diff, axis=-1)

    D_init = pairwise_dists(init_pos)
    D_final = pairwise_dists(final_pos)

    # 对每个中间 image 做几步 IDPP 弛豫
    for i in range(1, n_images - 1):
        t = ts[i]
        D_target = (1.0 - t) * D_init + t * D_final
        pos = images[i].copy()
        for _ in range(50):  # 50 步最速下降, 够收敛到 IDPP 最小
            diff = pos[:, None, :] - pos[None, :, :]  # (N, N, 3)
            r = np.linalg.norm(diff, axis=-1)  # (N, N)
            # 避免除零
            r_safe = np.where(r > 1e-10, r, 1e-10)
            D_target_safe = np.where(D_target > 1e-10, D_target, 1e-10)
            # 目标函数对 r_j 的梯度
            # f = sum_{j<k} (r_jk - D_jk)^2 / D_jk^2
            # df/dr_jk = 2 (r_jk - D_jk) / D_jk^2
            # dr_jk/dr_j = (r_j - r_k) / r_jk  (单位矢量 r_hat)
            coef = 2.0 * (r_safe - D_target_safe) / (D_target_safe ** 2)
            np.fill_diagonal(coef, 0.0)
            # r_hat[j, k, :] = (r_j - r_k) / r_jk
            r_hat = diff / r_safe[:, :, None]  # (N, N, 3)
            # grad[j, :] = sum_k coef[j, k] * r_hat[j, k, :]
            grad = np.sum(coef[:, :, None] * r_hat, axis=1)  # (N, 3)
            step = 0.05
            pos = pos - step * grad
        images[i] = pos

    return images


def compute_neb_forces(
    images: list[np.ndarray],
    true_forces: list[np.ndarray],
    energies: list[float],
    k: float,
    climbing_image: bool,
) -> list[np.ndarray]:
    """NEB 力投影: 切向真力 + 法向弹簧力.

    参考 Henkelman & Jónsson, J. Chem. Phys. 113, 9901 (2000) 的
    improved tangent estimate.
    """
    n = len(images)
    neb_forces: list[np.ndarray] = []

    # 端点不动
    neb_forces.append(np.zeros_like(true_forces[0]))
    neb_forces.append(np.zeros_like(true_forces[-1]))  # 占位, 末尾再放

    # 找最高能 image (CI-NEB)
    e_arr = np.asarray(energies)
    ci_idx = int(np.argmax(e_arr)) if climbing_image else -1

    for i in range(1, n - 1):
        F = true_forces[i]
        # improved tangent: 用相邻两 image 能量差定符号
        t_plus = images[i + 1] - images[i]
        t_minus = images[i] - images[i - 1]
        t_plus_norm = np.linalg.norm(t_plus)
        t_minus_norm = np.linalg.norm(t_minus)
        if t_plus_norm < 1e-10 or t_minus_norm < 1e-10:
            tau = t_plus if t_plus_norm >= t_minus_norm else t_minus
        else:
            # 能量加权切向 (Henkelman 2000 公式)
            e_prev = energies[i - 1]
            e_next = energies[i + 1]
            e_curr = energies[i]
            if e_next > e_curr > e_prev:
                tau = t_plus
            elif e_prev > e_curr > e_next:
                tau = t_minus
            else:
                # 中间情况: 用能量差加权
                dE_max = max(abs(e_next - e_curr), abs(e_prev - e_curr))
                dE_min = min(abs(e_next - e_curr), abs(e_prev - e_curr))
                if e_next > e_prev:
                    tau = dE_max * t_plus + dE_min * t_minus
                else:
                    tau = dE_min * t_plus + dE_max * t_minus
        tau_norm = np.linalg.norm(tau)
        if tau_norm < 1e-12:
            # 退化: 没法定切向, 干脆不动这个 image
            neb_forces.insert(-1, np.zeros_like(F))
            continue
        tau = tau / tau_norm

        # climbing image: 把平行分量翻转, 让它爬到鞍点
        if i == ci_idx:
            # F · τ 是全 3N 空间的标量投影, F 和 tau 都是 (N, 3)
            F_parallel = np.sum(F * tau) * tau
            F_neb = F - 2.0 * F_parallel
            neb_forces.insert(-1, F_neb)
            continue

        # 普通图像: F_perp (真力) + F_parallel_spring (弹簧力)
        F_perp = F - np.sum(F * tau) * tau

        # 弹簧力: 法向投影, 大小由相邻 image 距离差决定
        spring_mag = k * (
            np.linalg.norm(images[i + 1] - images[i])
            - np.linalg.norm(images[i] - images[i - 1])
        )
        F_spring = spring_mag * tau

        neb_forces.insert(-1, F_perp + F_spring)

    # 替换末尾占位
    neb_forces[-1] = np.zeros_like(true_forces[-1])
    return neb_forces


def compute_barriers(
    energies: list[float]
) -> tuple[float, float, int]:
    """算正反向势垒和鞍点 image 索引."""
    e_arr = np.asarray(energies, dtype=float)
    e_init = float(e_arr[0])
    e_final = float(e_arr[-1])
    saddle_idx = int(np.argmax(e_arr))
    e_saddle = float(e_arr[saddle_idx])
    forward = e_saddle - e_init
    reverse = e_saddle - e_final
    return forward, reverse, saddle_idx


def compute_path_length(images: list[np.ndarray]) -> float:
    """累计路径长度 (Å)."""
    total = 0.0
    for i in range(1, len(images)):
        total += float(np.linalg.norm(images[i] - images[i - 1]))
    return total
