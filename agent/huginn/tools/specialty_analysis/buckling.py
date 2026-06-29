"""特征值屈曲分析 — 解广义特征值问题 (K_G) φ = λ (K) φ.

参考:
- Bathe《Finite Element Procedures》Sec. 6.13 (屈曲特征值)
- Timoshenko & Gere《Theory of Elastic Stability》Ch. 2
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import eigh

from huginn.types import ToolResult


def eigenvalue_buckling(args: Any) -> ToolResult:
    """解 (K_G) φ = λ (K) φ, 返回前 N 阶临界载荷因子 λ_cr.

    λ_cr < 1 表示当前载荷下会发生屈曲; λ_cr > 1 表示安全裕度.
    """
    K = np.asarray(args.stiffness_matrix, dtype=float)
    KG = np.asarray(args.geometric_stiffness, dtype=float)
    n = K.shape[0]

    if K.shape != KG.shape:
        return ToolResult(
            data=None,
            success=False,
            error=(
                f"stiffness_matrix shape {K.shape} != geometric_stiffness shape {KG.shape}"
            ),
        )

    n_modes = min(args.num_modes, n)

    try:
        # eigh 解对称广义特征值问题 (K_G) φ = λ (K) φ
        # 返回升序特征值
        eigvals, eigvecs = eigh(KG, K, subset_by_index=[0, n_modes - 1])
    except np.linalg.LinAlgError as exc:
        return ToolResult(
            data=None,
            success=False,
            error=f"eigenvalue solve failed: {exc}",
        )

    # 过滤负特征值 (物理无意义) 和 NaN
    valid_mask = np.isfinite(eigvals) & (eigvals > 0)
    if not valid_mask.any():
        return ToolResult(
            data=None,
            success=False,
            error="no positive eigenvalues — check K_G sign convention (compressive load should give positive K_G)",
        )

    crit_load_factors = eigvals[valid_mask].tolist()
    mode_shapes = eigvecs[:, valid_mask].tolist()

    return ToolResult(
        data={
            "critical_load_factors": crit_load_factors,
            "first_critical_factor": crit_load_factors[0] if crit_load_factors else None,
            "mode_shapes": mode_shapes,
            "n_modes_returned": len(crit_load_factors),
            "matrix_dimension": n,
        },
        success=True,
    )
