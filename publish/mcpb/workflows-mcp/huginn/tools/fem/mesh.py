"""网格生成 — 矩形/圆形/立方体.

用 scikit-fem 的 MeshTri.init_tensor 构造结构化网格, 返回 Mesh + 边界 facet 标签.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from skfem import MeshTri

from huginn.core_types import ToolResult


def _build_mesh(shape: str, dims: dict[str, float], n_div: int) -> MeshTri | None:
    """按 shape 构造网格; 不支持的 shape 返回 None."""
    if shape == "rectangle":
        L = dims["L"]
        H = dims["H"]
        nx = max(n_div, 2)
        ny = max(n_div, 2)
        return MeshTri.init_tensor(
            np.linspace(0.0, L, nx + 1),
            np.linspace(0.0, H, ny + 1),
        )
    if shape == "circle":
        R = dims["R"]
        n_radial = max(n_div, 3)
        n_angular = max(n_div * 4, 12)
        # 极坐标网格
        r = np.linspace(0.0, R, n_radial + 1)
        theta = np.linspace(0.0, 2.0 * np.pi, n_angular + 1)
        # 在极坐标里 init_tensor 然后变换
        m = MeshTri.init_tensor(r, theta)
        # 极坐标 → 笛卡尔
        x = m.p[0] * np.cos(m.p[1])
        y = m.p[0] * np.sin(m.p[1])
        m.p = np.vstack([x, y])
        return m
    return None


def _boundary_facets(m: Any, dims: dict[str, float]) -> dict[str, Any]:
    """标记边界 facets (left/right/bottom/top)."""
    # scikit-fem MeshTri.facets_satisfying 返回 facet 索引
    try:
        left_facets = m.facets_satisfying(lambda x: x[0] < 1e-10)
        right_facets = m.facets_satisfying(
            lambda x: x[0] > dims.get("L", dims.get("R", 1.0)) - 1e-10
        )
        bottom_facets = m.facets_satisfying(lambda x: x[1] < 1e-10)
        top_facets = m.facets_satisfying(
            lambda x: x[1] > dims.get("H", dims.get("R", 1.0)) - 1e-10
        )
    except Exception:
        left_facets = right_facets = bottom_facets = top_facets = []
    return {
        "left": left_facets,
        "right": right_facets,
        "bottom": bottom_facets,
        "top": top_facets,
    }


def _collect_fixed_dofs(
    bcs: Any, boundary_facets: dict[str, Any], m: Any, basis: Any
) -> np.ndarray:
    """按 boundary_conditions 收集被固定的 DOF 索引."""
    fixed_dofs = []
    for bc in bcs:
        region = bc.get("region", "left")
        region_facets = boundary_facets.get(region, [])
        if len(region_facets):
            region_nodes = np.unique(m.facets[:, region_facets])
            for node in region_nodes:
                for dof_idx in bc.get("dofs", [0, 1]):
                    fixed_dofs.append(basis.nodal_dofs[dof_idx][node])
    return np.array(fixed_dofs, dtype=int) if fixed_dofs else np.array([], dtype=int)


def mesh_from_geometry(args: Any) -> ToolResult:
    """根据 dims 和 shape 构造 scikit-fem 网格.

    rectangle: {L, H} → nx × ny 四边形 (剖分为三角形)
    circle:    {R}    → 极坐标网格
    cube:      {L, W, H} → 3D (暂不支持, 返回错误)
    """
    shape = args.shape
    dims = args.dims
    n_div = args.n_div

    try:
        m = _build_mesh(shape, dims, n_div)
        if m is None:
            return ToolResult(
                data=None,
                success=False,
                error=f"shape '{shape}' not supported. Use 'rectangle' or 'circle'.",
            )
    except KeyError as exc:
        return ToolResult(
            data=None,
            success=False,
            error=f"missing dimension {exc} for shape '{shape}'",
        )
    except Exception as exc:
        return ToolResult(
            data=None,
            success=False,
            error=f"mesh generation failed: {exc}",
        )

    return ToolResult(
        data={
            "mesh": m,
            "n_nodes": m.p.shape[1],
            "n_elements": m.t.shape[1],
            "shape": shape,
            "dims": dims,
            "n_div": n_div,
            "boundary_facets": _boundary_facets(m, dims),
        },
        success=True,
    )
