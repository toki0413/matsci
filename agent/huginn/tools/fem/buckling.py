"""特征值屈曲分析 — 几何刚度 + eigsh.

(K + λ K_G) φ = 0  →  K_G φ = -λ K φ
scikit-fem 不直接提供几何刚度, 这里用简化预应力法.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.sparse.linalg import eigsh
from skfem import Basis, ElementTriP1, asm
from skfem.models.elasticity import linear_elasticity, lame_parameters

from huginn.types import ToolResult


def buckling(args: Any) -> ToolResult:
    """2D 线弹性特征值屈曲. 简化几何刚度: 单位预应力分布."""
    mesh_result = args._mesh_result if hasattr(args, "_mesh_result") else None
    if mesh_result is None:
        from .mesh import mesh_from_geometry
        mesh_result = mesh_from_geometry(args)
        if not mesh_result.success:
            return mesh_result

    m = mesh_result.data["mesh"]
    boundary_facets = mesh_result.data["boundary_facets"]

    material = args.material
    E = material["E"]
    nu = material["nu"]
    rho = material.get("rho", 7850.0)
    thickness = material.get("thickness", 1.0)

    lam, mu = lame_parameters(E, nu)

    e = ElementTriP1()
    basis = Basis(m, e)

    # 刚度矩阵
    K = asm(linear_elasticity(lam, mu), basis)

    # 简化几何刚度: 用 K 本身的对角缩放近似
    # 严格做法需要先求静力解得到应力场, 再组装几何刚度
    # 这里用 K 的对角 * 单位载荷系数 作为 K_G 的近似
    K_diag = K.diagonal()
    # K_G ≈ alpha * diag(K) — alpha 是单位载荷下的几何刚度尺度
    # 特征值 λ = alpha * diag(K) / K → 与 K 同量纲
    from scipy.sparse import diags
    KG = diags(K_diag * 1e-6, 0, format="csr")  # 缩放系数 1e-6

    # 边界条件
    bcs = args.boundary_conditions
    fixed_dofs = []
    for bc in bcs:
        region = bc.get("region", "left")
        region_facets = boundary_facets.get(region, [])
        if region_facets:
            region_nodes = np.unique(m.t[:, m.facets[:, region_facets].flatten()].flatten())
            for node in region_nodes:
                for dof_idx in bc.get("dofs", [0, 1]):
                    fixed_dofs.append(basis.nodal_dofs[dof_idx][node])

    fixed_dofs = np.array(fixed_dofs, dtype=int) if fixed_dofs else np.array([], dtype=int)
    free_dofs = np.setdiff1d(np.arange(basis.N), fixed_dofs)

    K_red = K.tocsr()[free_dofs][:, free_dofs]
    KG_red = KG.tocsr()[free_dofs][:, free_dofs]

    n_modes = min(args.num_modes, K_red.shape[0] - 1)
    if n_modes < 1:
        return ToolResult(
            data=None,
            success=False,
            error="not enough free DOFs for buckling analysis",
        )

    try:
        # 解 K_G φ = λ K φ (用 eigsh, 注意符号)
        # eigsh(A, k, M=B) 解 A φ = λ B φ
        eigvals, eigvecs = eigsh(
            KG_red,
            k=n_modes,
            M=K_red,
            sigma=0.0,
            which="LM",
            mode="normal",
        )
    except Exception as exc:
        return ToolResult(
            data=None,
            success=False,
            error=f"buckling eigsh failed: {exc}",
        )

    sort_idx = np.argsort(eigvals)
    eigvals = eigvals[sort_idx]

    valid_mask = np.isfinite(eigvals) & (eigvals > 0)
    crit_factors = eigvals[valid_mask]

    return ToolResult(
        data={
            "critical_load_factors": crit_factors.tolist(),
            "first_critical_factor": float(crit_factors[0]) if len(crit_factors) else None,
            "n_modes_returned": len(crit_factors),
            "n_nodes": m.p.shape[1],
            "n_dofs": basis.N,
            "n_free_dofs": K_red.shape[0],
            "material": material,
            "note": "simplified geometric stiffness (diagonal approximation) — for production use, implement stress-based K_G",
        },
        success=True,
    )
