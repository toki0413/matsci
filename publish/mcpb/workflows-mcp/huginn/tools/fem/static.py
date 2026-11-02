"""线性静力分析 — 平面应力/应变.

K u = f, K 由 scikit-fem linear_elasticity 组装.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from skfem import (
    Basis,
    ElementTriP2,
    ElementVector,
    asm,
    condense,
    solve,
)
from skfem.models.elasticity import linear_elasticity, plane_stress

from huginn.core_types import ToolResult


def _assemble_stiffness(m: Any, material: dict[str, float]) -> tuple[Any, Any, float]:
    """组装刚度矩阵 K, 返回 (basis, K, thickness)."""
    E = material["E"]
    nu = material["nu"]
    material.get("rho", 7850.0)
    thickness = material.get("thickness", 1.0)

    # plane_stress gives the correct 2D Lame parameters for plane stress analysis
    lam, mu = plane_stress(E, nu)

    # P2 (quadratic) elements: P1 is too stiff for bending and gives <50% of analytical
    e = ElementVector(ElementTriP2())
    basis = Basis(m, e)

    # 组装刚度矩阵
    K = asm(linear_elasticity(lam, mu), basis)
    return basis, K, thickness


def _assemble_loads(
    loads: Any, boundary_facets: dict[str, Any], m: Any, basis: Any, thickness: float
) -> np.ndarray:
    """组装载荷向量: 支持端部集中力 (right edge) 和顶部均布压力 (top edge)."""
    f = np.zeros(basis.N)

    for load in loads:
        ltype = load.get("type", "point")
        value = float(load.get("value", 0.0))
        region = load.get("region", "right")

        if ltype == "point":
            # 集中力: 在 region 边界节点上加力
            region_facets = boundary_facets.get(region, [])
            if len(region_facets):
                region_nodes = np.unique(m.facets[:, region_facets])
                for node in region_nodes:
                    dofs_y = basis.nodal_dofs[1][node : node + 1]
                    f[dofs_y] += value / max(len(region_nodes), 1)

        elif ltype == "pressure":
            # 均布压力: 用 LinearForm 在 region 边界积分
            region_facets = boundary_facets.get(region, [])
            if len(region_facets):
                region_nodes = np.unique(m.facets[:, region_facets])
                for node in region_nodes:
                    dofs_y = basis.nodal_dofs[1][node : node + 1]
                    f[dofs_y] += value * thickness / max(len(region_nodes), 1)
    return f


def _displacement_result(
    u: Any, basis: Any, m: Any, material: dict[str, float], loads: Any, bcs: Any
) -> ToolResult:
    """从解向量提取位移场并组装返回数据."""
    # 提取位移
    ux = u[basis.nodal_dofs[0]]
    uy = u[basis.nodal_dofs[1]]
    u_mag = np.sqrt(ux**2 + uy**2)

    max_disp_idx = int(np.argmax(u_mag))
    max_displacement = float(u_mag[max_disp_idx])
    max_disp_location = [float(m.p[0][max_disp_idx]), float(m.p[1][max_disp_idx])]

    return ToolResult(
        data={
            "max_displacement": max_displacement,
            "max_displacement_location": max_disp_location,
            "displacement_x": ux.tolist(),
            "displacement_y": uy.tolist(),
            "displacement_magnitude": u_mag.tolist(),
            "n_nodes": m.p.shape[1],
            "n_dofs": basis.N,
            "material": material,
            "loads": loads,
            "boundary_conditions": bcs,
        },
        success=True,
    )


def static_linear(args: Any) -> ToolResult:
    """2D 线弹性静力分析. 接受 mesh_from_geometry 的输出或内部重新生成."""
    mesh_result = args._mesh_result if hasattr(args, "_mesh_result") else None
    if mesh_result is None:
        # 内部调 mesh_from_geometry
        from .mesh import mesh_from_geometry

        mesh_result = mesh_from_geometry(args)
        if not mesh_result.success:
            return mesh_result

    m = mesh_result.data["mesh"]
    boundary_facets = mesh_result.data["boundary_facets"]

    material = args.material
    loads = args.loads
    basis, K, thickness = _assemble_stiffness(m, material)
    f = _assemble_loads(loads, boundary_facets, m, basis, thickness)

    # 边界条件: 固定 left edge (u1=u2=0)
    from .mesh import _collect_fixed_dofs

    bcs = args.boundary_conditions
    fixed_dofs = _collect_fixed_dofs(bcs, boundary_facets, m, basis)

    # 求解 K u = f (缩减后)
    try:
        u = solve(*condense(K, f, D=fixed_dofs))
    except Exception as exc:
        return ToolResult(
            data=None,
            success=False,
            error=f"linear solve failed: {exc}",
        )
    return _displacement_result(u, basis, m, material, loads, bcs)
