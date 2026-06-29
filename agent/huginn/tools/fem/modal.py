"""模态分析 — 一致质量矩阵 + eigsh.

K φ = ω² M φ, K 和 M 由 scikit-fem 组装.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.sparse.linalg import eigsh
from skfem import Basis, BilinearForm, ElementTriP1, asm
from skfem.models.elasticity import linear_elasticity, lame_parameters

from huginn.types import ToolResult


def modal(args: Any) -> ToolResult:
    """2D 线弹性模态分析. 返回前 N 阶固有频率."""
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

    # 一致质量矩阵: M = ∫ ρ N^T N dV
    @BilinearForm
    def mass_form(u, v, w):
        return rho * thickness * (u[0] * v[0] + u[1] * v[1])

    M = asm(mass_form, basis)

    # 边界条件: 固定 left edge
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

    # 缩减矩阵 (去掉固定 DOF)
    free_dofs = np.setdiff1d(np.arange(basis.N), fixed_dofs)
    K_red = K.tocsr()[free_dofs][:, free_dofs]
    M_red = M.tocsr()[free_dofs][:, free_dofs]

    n_modes = min(args.num_modes, K_red.shape[0] - 1)
    if n_modes < 1:
        return ToolResult(
            data=None,
            success=False,
            error="not enough free DOFs for modal analysis",
        )

    try:
        # shift-invert Lanczos, sigma=0 求低阶模态
        eigvals, eigvecs = eigsh(
            K_red,
            k=n_modes,
            M=M_red,
            sigma=0.0,
            which="LM",
            mode="normal",
        )
    except Exception as exc:
        return ToolResult(
            data=None,
            success=False,
            error=f"eigsh failed: {exc}",
        )

    # 升序排序
    sort_idx = np.argsort(eigvals)
    eigvals = eigvals[sort_idx]

    # 过滤负/NaN
    valid_mask = np.isfinite(eigvals) & (eigvals > 0)
    omega_sq = eigvals[valid_mask]
    omega = np.sqrt(omega_sq)
    freq_hz = omega / (2.0 * np.pi)

    return ToolResult(
        data={
            "angular_frequencies_rad_s": omega.tolist(),
            "frequencies_hz": freq_hz.tolist(),
            "first_natural_frequency_hz": float(freq_hz[0]) if len(freq_hz) else None,
            "n_modes_returned": len(omega),
            "n_nodes": m.p.shape[1],
            "n_dofs": basis.N,
            "n_free_dofs": K_red.shape[0],
            "material": material,
        },
        success=True,
    )
