"""模态分析 — shift-invert Lanczos 解广义特征值 (K) φ = ω² (M) φ.

参考:
- Bathe《Finite Element Procedures》Sec. 10.4 (Lanczos 算法)
- Hughes《The Finite Element Method》Sec. 7.3 (一致质量矩阵)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

from huginn.types import ToolResult


def modal_lanczos(args: Any) -> ToolResult:
    """解 (K) φ = ω² (M) φ, 返回前 N 阶固有频率.

    小矩阵 (n <= 200) 直接用 scipy.linalg.eigh (dense LAPACK).
    大矩阵用 scipy.sparse.linalg.eigsh + shift-invert (sigma) 求低阶模态.
    """
    K = np.asarray(args.stiffness_matrix, dtype=float)
    M = np.asarray(args.mass_matrix, dtype=float)
    n = K.shape[0]

    if K.shape != M.shape:
        return ToolResult(
            data=None,
            success=False,
            error=f"stiffness_matrix shape {K.shape} != mass_matrix shape {M.shape}",
        )

    # dense path 允许 n_modes <= n; sparse (eigsh) 要求 k < n
    if n <= 200:
        n_modes = min(args.num_modes, n)
    else:
        n_modes = min(args.num_modes, n - 1)
    n_modes = max(n_modes, 1)
    shift = args.shift if args.shift is not None else 0.0

    try:
        if n <= 200:
            # dense 求解 — 直接拿前 N 阶
            eigvals, eigvecs = eigh(K, M, subset_by_index=[0, n_modes - 1])
        else:
            # 大矩阵用 shift-invert Lanczos, sigma=shift 求靠近 shift 的特征值
            # which='LM' on (K - σM)^{-1} M 等价于求靠近 σ 的特征值
            eigvals, eigvecs = eigsh(
                K,
                k=n_modes,
                M=M,
                sigma=shift,
                which="LM",
                mode="normal",
            )
            # 升序排序
            sort_idx = np.argsort(eigvals)
            eigvals = eigvals[sort_idx]
            eigvecs = eigvecs[:, sort_idx]
    except np.linalg.LinAlgError as exc:
        return ToolResult(
            data=None,
            success=False,
            error=f"eigenvalue solve failed: {exc}",
        )
    except Exception as exc:
        # eigsh 在矩阵奇异时会抛 RuntimeError
        return ToolResult(
            data=None,
            success=False,
            error=f"Lanczos solver failed (try different shift): {exc}",
        )

    # 过滤负 / NaN 特征值 (前几阶可能是接近刚体模态的小负数)
    valid_mask = np.isfinite(eigvals) & (eigvals > 0)
    if not valid_mask.any():
        return ToolResult(
            data=None,
            success=False,
            error="no positive eigenvalues — check boundary conditions (rigid body modes?)",
        )

    omega_sq = eigvals[valid_mask]
    omega = np.sqrt(omega_sq)
    freq_hz = omega / (2.0 * np.pi)

    mode_shapes = eigvecs[:, valid_mask].tolist()

    return ToolResult(
        data={
            "angular_frequencies_rad_s": omega.tolist(),
            "frequencies_hz": freq_hz.tolist(),
            "first_natural_frequency_hz": freq_hz[0] if len(freq_hz) else None,
            "mode_shapes": mode_shapes,
            "n_modes_returned": len(omega),
            "matrix_dimension": n,
            "solver": "dense_eigh" if n <= 200 else "lanczos_shift_invert",
        },
        success=True,
    )
