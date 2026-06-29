"""PES 地形拓扑分析 — TDA 互调 + basin 分析 + 网格极值检测.

tda_tool 注册时就调它算 Betti 数, 没注册就退到内置 basin 分析.
网格极值 (find_local_minima/saddles/extrema/count_connected_basins)
是 basin_analysis 的零件, 也可被 pes_scan 直接用.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from huginn.types import ToolContext, ToolResult

if TYPE_CHECKING:
    pass


async def topology_via_tda(
    pes: dict, energies: np.ndarray, context: ToolContext
) -> ToolResult:
    """调 tda_tool.energy_landscape_topology 算 Betti 数."""
    try:
        from huginn.tools.registry import ToolRegistry

        tda = ToolRegistry.get("tda_tool") or ToolRegistry.get("tda")
    except Exception:
        tda = None

    if tda is None:
        # tda_tool 没注册, 退到 basin 分析
        return basin_analysis(pes, energies)

    # 把网格能量展平成 (structures, energies)
    grid_axes = pes.get("grid_axes") or []
    scan_coords = pes.get("scan_coords") or []
    n_dims = pes.get("n_dims", 1)

    # 构造采样点坐标 (供 tda 算 spatial threshold)
    if grid_axes and scan_coords:
        mesh = np.meshgrid(*[np.asarray(a) for a in grid_axes[:n_dims]], indexing="ij")
        pts = np.stack([m.ravel() for m in mesh], axis=1)
        # 把扫描坐标扩展成 3N 维结构空间 (其他原子位置不动, 只在 scan 坐标上动)
        # 这里简化: 直接用扫描坐标作为 structures 输入给 tda
        structures = pts.tolist()
    else:
        structures = [[float(i)] for i in range(energies.size)]

    flat_energies = energies.ravel().tolist()

    try:
        tda_input = tda.input_schema(
            action="energy_landscape_topology",
            structures=structures,
            energies=flat_energies,
        )
        result = await tda.call(tda_input, context)
        if not result.success:
            return basin_analysis(pes, energies)

        data = result.data or {}
        return ToolResult(
            data={
                "action": "landscape_topology",
                "method": "tda",
                "betti_numbers": {
                    "betti_0": int(data.get("n_basins", 0)),
                    "betti_1": int(data.get("n_pathways", 0)),
                },
                "persistence_diagram": data.get("persistence_diagram", []),
                "n_minima": int(data.get("n_basins", 0)),
                "n_saddles": int(data.get("n_pathways", 0)),
                "connectivity": float(data.get("connectivity", 0.0)),
                "energy_threshold": float(data.get("energy_threshold", 0.0)),
                "spatial_threshold": float(data.get("spatial_threshold", 0.0)),
                "n_structures": int(data.get("n_structures", 0)),
                "n_edges": int(data.get("n_edges", 0)),
            },
            success=True,
        )
    except Exception as exc:
        # tda 调用挂了, 退到内置 basin 分析
        return basin_analysis(pes, energies)


def basin_analysis(
    pes: dict, energies: np.ndarray
) -> ToolResult:
    """简易 basin 分析: 按能量阈值分层, 数极小和鞍点.

    这里用网格上的局部极值 + 阈值连通性近似 Betti 数, 不依赖 scipy.
    """
    flat = energies.ravel()
    e_min = float(np.min(flat))
    e_max = float(np.max(flat))
    # 阈值取能量区间 1/3 处, 把"低谷"分出来
    e_thr = e_min + 0.33 * (e_max - e_min)

    # 用扫描范围里的极值点近似极小 / 鞍点
    minima_pts = find_local_minima(energies)
    saddle_pts = find_local_saddles(energies)

    # Betti-0 近似 = 低于阈值的连通分量数 (4-邻接连通, BFS)
    n_basins = count_connected_basins(energies, e_thr)
    # Betti-1 近似 = 网格上独立"环路"数, 用 E - V + C 估算
    n_low = int(np.sum(flat <= e_thr))
    # 简化: 在二维网格上 betti_1 ≈ 极小数 - 连通分量数; 一维直接 0
    n_saddles = max(0, len(saddle_pts))
    betti_1 = max(0, len(minima_pts) - n_basins) if energies.ndim == 2 else 0

    return ToolResult(
        data={
            "action": "landscape_topology",
            "method": "basin_analysis",
            "betti_numbers": {
                "betti_0": int(n_basins),
                "betti_1": int(betti_1),
            },
            "persistence_diagram": [],
            "n_minima": int(len(minima_pts)),
            "n_saddles": int(n_saddles),
            "connectivity": float(
                n_low / max(1, flat.size)
            ),
            "energy_threshold": float(e_thr),
            "minima": minima_pts,
            "saddle_points": saddle_pts,
            "energy_range": [e_min, e_max],
        },
        success=True,
    )


def find_local_minima(grid: np.ndarray) -> list[dict]:
    """网格上的局部极小点 (4/6 邻域)."""
    ndim = grid.ndim
    minima: list[dict] = []
    for idx in np.ndindex(*grid.shape):
        val = grid[idx]
        is_min = True
        for ax in range(ndim):
            for off in (-1, 1):
                nb = list(idx)
                nb[ax] += off
                if 0 <= nb[ax] < grid.shape[ax]:
                    if grid[tuple(nb)] < val:
                        is_min = False
                        break
            if not is_min:
                break
        if is_min:
            minima.append({"index": list(idx), "energy": float(val)})
    return minima


def find_local_saddles(grid: np.ndarray) -> list[dict]:
    """网格上的鞍点近似: 一维上局部极大, 二维上鞍点 (一维极大一维极小)."""
    saddles: list[dict] = []
    if grid.ndim == 1:
        # 一维: 极大值点当作"过渡态"
        for i in range(1, grid.shape[0] - 1):
            if grid[i] > grid[i - 1] and grid[i] > grid[i + 1]:
                saddles.append({"index": [i], "energy": float(grid[i])})
    elif grid.ndim == 2:
        # 二维: 鞍点 = 一维极大 + 另一维极小
        for i in range(1, grid.shape[0] - 1):
            for j in range(1, grid.shape[1] - 1):
                v = grid[i, j]
                max_x = v > grid[i - 1, j] and v > grid[i + 1, j]
                min_y = v < grid[i, j - 1] and v < grid[i, j + 1]
                min_x = v < grid[i - 1, j] and v < grid[i + 1, j]
                max_y = v > grid[i, j - 1] and v > grid[i, j + 1]
                if (max_x and min_y) or (min_x and max_y):
                    saddles.append(
                        {"index": [i, j], "energy": float(v)}
                    )
    return saddles


def find_extrema(
    energies: np.ndarray,
    mesh: list[np.ndarray],
    scan_coords: list[list[int]],
) -> tuple[list[dict], list[dict]]:
    """给 PES 扫描结果打包极小 / 鞍点, 带上扫描坐标值."""
    minima_raw = find_local_minima(energies)
    saddles_raw = find_local_saddles(energies)

    def _annotate(item: dict) -> dict:
        idx = item["index"]
        coords = []
        for d, (atom_idx, axis) in enumerate(scan_coords):
            if d < len(mesh):
                coords.append(
                    {
                        "atom_idx": int(atom_idx),
                        "axis": int(axis),
                        "displacement": float(mesh[d][tuple(idx)]),
                    }
                )
        item["scan_coords"] = coords
        return item

    minima = [_annotate(m) for m in minima_raw]
    saddles = [_annotate(s) for s in saddles_raw]
    return minima, saddles


def count_connected_basins(
    grid: np.ndarray, threshold: float
) -> int:
    """数低于 threshold 的网格点的连通分量数 (4/6 邻域, BFS)."""
    mask = grid <= threshold
    visited = np.zeros_like(mask, dtype=bool)
    n_comp = 0
    for start in np.ndindex(*grid.shape):
        if mask[start] and not visited[start]:
            # BFS
            queue = [start]
            visited[start] = True
            while queue:
                cur = queue.pop()
                for ax in range(grid.ndim):
                    for off in (-1, 1):
                        nb = list(cur)
                        nb[ax] += off
                        if (
                            0 <= nb[ax] < grid.shape[ax]
                            and mask[tuple(nb)]
                            and not visited[tuple(nb)]
                        ):
                            visited[tuple(nb)] = True
                            queue.append(tuple(nb))
            n_comp += 1
    return n_comp
