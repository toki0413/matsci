"""模态分析 — 一致质量矩阵 + eigsh.

K φ = ω² M φ, K 和 M 由 scikit-fem 组装.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.sparse.linalg import eigsh
from skfem import Basis, BilinearForm, ElementTriP2, ElementVector, asm
from skfem.models.elasticity import linear_elasticity, plane_stress

from huginn.core_types import ToolResult


def _assemble_modal_system(m: Any, material: dict[str, float]) -> tuple[Any, Any, Any]:
    """组装刚度矩阵 K 和一致质量矩阵 M."""
    E = material["E"]
    nu = material["nu"]
    rho = material.get("rho", 7850.0)
    thickness = material.get("thickness", 1.0)

    lam, mu = plane_stress(E, nu)

    e = ElementVector(ElementTriP2())
    basis = Basis(m, e)

    # 刚度矩阵
    K = asm(linear_elasticity(lam, mu), basis)

    # 一致质量矩阵: M = ∫ ρ N^T N dV
    @BilinearForm
    def mass_form(u, v, w):
        return rho * thickness * (u[0] * v[0] + u[1] * v[1])

    M = asm(mass_form, basis)
    return basis, K, M


def _solve_modal_eigsh(
    K_red: Any, M_red: Any, n_modes: int
) -> tuple[Any, ToolResult | None]:
    """解 K φ = ω² M φ; 返回 (eigvals, 错误结果或 None)."""
    try:
        # shift-invert Lanczos, sigma=0 求低阶模态
        eigvals, _eigvecs = eigsh(
            K_red,
            k=n_modes,
            M=M_red,
            sigma=0.0,
            which="LM",
            mode="normal",
        )
    except Exception as exc:
        return None, ToolResult(
            data=None,
            success=False,
            error=f"eigsh failed: {exc}",
        )
    return eigvals, None


def _modal_result(
    eigvals: Any, m: Any, basis: Any, K_red: Any, material: dict[str, float]
) -> ToolResult:
    """换算角频率/频率并组装返回数据."""
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
    basis, K, M = _assemble_modal_system(m, material)

    # 边界条件: 固定 left edge
    from .mesh import _collect_fixed_dofs

    fixed_dofs = _collect_fixed_dofs(
        args.boundary_conditions, boundary_facets, m, basis
    )

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

    eigvals, err = _solve_modal_eigsh(K_red, M_red, n_modes)
    if err is not None:
        return err
    return _modal_result(eigvals, m, basis, K_red, material)
