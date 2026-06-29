"""缺陷检测 — 裂纹 / 气孔 / 夹杂, 基于 scipy.ndimage 形态学."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from huginn.types import ToolResult
from huginn.tools.image_analysis._utils import load_gray, otsu_numpy

if TYPE_CHECKING:
    from huginn.tools.image_analysis.tool import ImageAnalysisInput

logger = logging.getLogger(__name__)


def defect_detect(args: "ImageAnalysisInput") -> ToolResult:
    arr = load_gray(args.image_path)
    defect_type = args.parameters.get("defect_type", "crack")
    sensitivity = float(args.parameters.get("sensitivity", 0.5))
    pixel_size = float(args.parameters.get("pixel_size_nm", 1.0))
    min_defect_area = int(args.parameters.get("min_defect_area_px", 5))

    try:
        from scipy.ndimage import (
            binary_closing,
            binary_opening,
            generate_binary_structure,
            grey_opening,
            label as nd_label,
        )
    except ImportError as exc:
        return ToolResult(
            data=None,
            success=False,
            error=f"缺陷检测需要 scipy.ndimage, 请安装 scipy ({exc})",
        )

    # sensitivity 越高, 阈值越宽松
    sens = max(0.0, min(1.0, sensitivity))
    structure = generate_binary_structure(2, 1)

    if defect_type == "crack":
        # 裂纹: 用 grey top-hat 找长条暗结构
        se_size = max(3, int(20 * (1.0 - sens)))
        bg = grey_opening(arr, size=se_size)
        tophat_neg = bg - arr  # 暗残差
        thr_val = float(np.percentile(tophat_neg, 95 - 10 * sens))
        binary = tophat_neg > thr_val
        binary = binary_opening(binary, structure=structure)
        defect_label = "crack"
    elif defect_type == "pore":
        # 气孔: 暗圆形 blob
        otsu_t = otsu_numpy(arr)
        thr_val = otsu_t - 10.0 * (1.0 - sens)
        binary = arr < thr_val
        binary = binary_closing(binary, structure=structure, iterations=2)
        binary = binary_opening(binary, structure=structure, iterations=1)
        defect_label = "pore"
    elif defect_type == "inclusion":
        # 夹杂: 亮的高衬度区
        otsu_t = otsu_numpy(arr)
        thr_val = otsu_t + 30.0 * (1.0 - sens)
        binary = arr > thr_val
        binary = binary_closing(binary, structure=structure, iterations=2)
        binary = binary_opening(binary, structure=structure, iterations=1)
        defect_label = "inclusion"
    else:
        return ToolResult(
            data=None,
            success=False,
            error=f"未知缺陷类型: {defect_type}, 支持 crack/pore/inclusion",
        )

    labeled, n_defects = nd_label(binary)
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    areas_all = counts[counts > 0]
    areas_f = areas_all[areas_all >= min_defect_area]

    defects_info: list[dict[str, Any]] = []
    for i in range(1, n_defects + 1):
        a = int(counts[i])
        if a < min_defect_area:
            continue
        ys, xs = np.where(labeled == i)
        info: dict[str, Any] = {
            "id": i,
            "area_px2": a,
            "area_nm2": float(a * pixel_size ** 2),
            "centroid_px": [float(xs.mean()), float(ys.mean())],
            "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        }
        if defect_type == "crack":
            w = int(xs.max() - xs.min() + 1)
            h = int(ys.max() - ys.min() + 1)
            info["aspect_ratio"] = float(max(w, h) / max(min(w, h), 1))
        defects_info.append(info)
        if len(defects_info) >= 100:
            break

    if len(areas_f) > 0:
        summary = (
            f"缺陷检测 ({defect_label}): 检测到 {len(defects_info)} 个候选缺陷, "
            f"平均面积 {float(areas_f.mean()):.1f} px²"
        )
    else:
        summary = f"缺陷检测 ({defect_label}): 未检测到符合要求的缺陷"

    data = {
        "summary": summary,
        "measurements": {
            "image_shape": [int(arr.shape[0]), int(arr.shape[1])],
            "defect_type": defect_label,
            "sensitivity": sens,
            "n_defects": int(len(defects_info)),
            "defects": defects_info,
            "total_defect_area_px2": int(areas_f.sum()),
            "defect_area_fraction": float(
                areas_f.sum() / (arr.shape[0] * arr.shape[1])
            ),
        },
    }
    return ToolResult(data=data)
