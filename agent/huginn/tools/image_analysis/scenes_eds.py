"""EDS 元素 mapping — 颜色匹配 + 覆盖率 + 元素重叠分析."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from huginn.types import ToolResult
from huginn.tools.image_analysis._utils import load_rgb, parse_color, auto_detect_colors

if TYPE_CHECKING:
    from huginn.tools.image_analysis.tool import ImageAnalysisInput

logger = logging.getLogger(__name__)


def eds_mapping(args: "ImageAnalysisInput") -> ToolResult:
    rgb = load_rgb(args.image_path)
    H, W, _ = rgb.shape

    element_colors_in = args.parameters.get("element_colors", None)
    overlap_threshold = float(args.parameters.get("overlap_threshold", 0.30))
    color_tolerance = float(args.parameters.get("color_tolerance", 30.0))

    if element_colors_in:
        element_colors = {
            str(elem): parse_color(c)
            for elem, c in element_colors_in.items()
        }
    else:
        # 没给颜色就自动 k-means 兜底
        element_colors = auto_detect_colors(rgb, k=5)

    element_masks: dict[str, np.ndarray] = {}
    element_stats: dict[str, Any] = {}
    for elem, target in element_colors.items():
        dist = np.sqrt(((rgb - target) ** 2).sum(axis=2))
        mask = dist < color_tolerance
        element_masks[elem] = mask
        n_px = int(mask.sum())
        coverage = float(n_px / (H * W))
        if n_px > 0:
            ys, xs = np.where(mask)
            cx = float(xs.mean())
            cy = float(ys.mean())
            spread_x = float(xs.std())
            spread_y = float(ys.std())
        else:
            cx = cy = spread_x = spread_y = 0.0
        element_stats[elem] = {
            "color_rgb": [int(target[0]), int(target[1]), int(target[2])],
            "coverage_fraction": coverage,
            "centroid_px": [cx, cy],
            "spread_px": [spread_x, spread_y],
            "n_pixels": n_px,
            "hotspots": _find_hotspots(mask, top_n=3),
        }

    # 两两元素的重叠
    overlaps: dict[str, Any] = {}
    elems = list(element_masks.keys())
    for i in range(len(elems)):
        for j in range(i + 1, len(elems)):
            e1, e2 = elems[i], elems[j]
            m1 = element_masks[e1]
            m2 = element_masks[e2]
            inter = int((m1 & m2).sum())
            union = int((m1 | m2).sum())
            iou = float(inter / union) if union > 0 else 0.0
            if iou > overlap_threshold:
                overlaps[f"{e1}+{e2}"] = {
                    "intersection_px": inter,
                    "iou": iou,
                }

    cov_str = ", ".join(
        f"{e}: {s['coverage_fraction'] * 100:.1f}%" for e, s in element_stats.items()
    )
    summary = f"EDS mapping: 检测到 {len(element_stats)} 种元素/相, {cov_str}"

    data = {
        "summary": summary,
        "measurements": {
            "image_shape": [int(H), int(W)],
            "color_tolerance": color_tolerance,
            "overlap_threshold": overlap_threshold,
            "elements": element_stats,
            "overlaps": overlaps,
        },
    }
    return ToolResult(data=data)


def _find_hotspots(mask: np.ndarray, top_n: int = 3) -> list[dict[str, Any]]:
    """找 mask 的 top-N 连通域 (热点区域), 返回质心 + 面积.

    EDS mapping 里元素常聚集成域, 整体质心不够精确. 找 top-N domain 让
    visual primitives 能指向具体聚集区, text-only LLM 能推理元素空间分布.
    ponytail: 复用 scipy.ndimage.label, 没装就返回空. 升级: DBSCAN + 密度峰.
    """
    try:
        from scipy.ndimage import label as nd_label, center_of_mass
    except ImportError:
        return []

    if int(mask.sum()) == 0:
        return []

    labeled, n_dom = nd_label(mask)
    if n_dom == 0:
        return []

    counts = np.bincount(labeled.ravel())
    counts[0] = 0  # 跳过背景 (label=0)
    valid_labels = np.where(counts > 0)[0]
    if len(valid_labels) == 0:
        return []

    # 按面积排序, 取 top-N
    sorted_labels = valid_labels[np.argsort(-counts[valid_labels])]
    top_labels = sorted_labels[:top_n]

    out: list[dict[str, Any]] = []
    for lbl in top_labels:
        area = int(counts[lbl])
        # center_of_mass 返回 (row, col) = (y, x)
        com = center_of_mass(mask, labeled, lbl)
        out.append({
            "centroid_px": [float(com[1]), float(com[0])],  # [x, y]
            "area_px2": area,
        })
    return out
