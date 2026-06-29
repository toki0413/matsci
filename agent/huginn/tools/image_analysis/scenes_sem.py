"""SEM 形貌分析 — 衬度统计 / 粗糙度 / 暗区分割 / 边缘密度."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from huginn.types import ToolResult
from huginn.tools.image_analysis._utils import load_gray

if TYPE_CHECKING:
    from huginn.tools.image_analysis.tool import ImageAnalysisInput

logger = logging.getLogger(__name__)


def sem_analysis(args: "ImageAnalysisInput") -> ToolResult:
    arr = load_gray(args.image_path)
    pixel_size = float(args.parameters.get("pixel_size_nm", 1.0))
    contrast_threshold = args.parameters.get("contrast_threshold", None)

    mean_i = float(arr.mean())
    std_i = float(arr.std())
    p5, p50, p95 = np.percentile(arr, [5, 50, 95])

    # 表面粗糙度: 用 uniform_filter 算局部均值, 残差的 RMS
    roughness_rms = std_i
    try:
        from scipy.ndimage import uniform_filter

        local_mean = uniform_filter(arr, size=15)
        residual = arr - local_mean
        roughness_rms = float(np.sqrt((residual ** 2).mean()))
    except ImportError:
        logger.debug("scipy 不可用, 粗糙度退化为全局 std")

    # 阈值分割找暗区 (颗粒/孔洞)
    if contrast_threshold is None:
        thr = mean_i - 0.5 * std_i
    else:
        thr = float(contrast_threshold)
    binary = arr < thr

    # 连通域统计
    areas: list[float] = []
    n_regions = 0
    try:
        from scipy.ndimage import label as nd_label

        labeled, n_regions = nd_label(binary)
        counts = np.bincount(labeled.ravel())
        counts[0] = 0
        areas = counts[counts > 0].astype(float).tolist()
    except ImportError:
        n_regions = int(binary.sum())
        logger.debug("scipy 不可用, 仅返回暗像素总数")

    # 用 cv2 跑一遍边缘统计作为可选增强
    edge_density = 0.0
    try:
        import cv2

        edges = cv2.Canny(arr.astype(np.uint8), 50, 150)
        edge_density = float(edges.sum() / 255 / edges.size)
    except ImportError:
        # numpy Sobel 兜底
        try:
            from scipy.ndimage import sobel

            gx = sobel(arr, axis=1)
            gy = sobel(arr, axis=0)
            grad = np.sqrt(gx ** 2 + gy ** 2)
            edge_density = float((grad > grad.mean() + 2 * grad.std()).mean())
        except ImportError:
            pass

    # 直方图 (32 bins)
    hist, _ = np.histogram(arr, bins=32, range=(0, 255))

    mean_area = float(np.mean(areas)) if areas else 0.0
    summary = (
        f"SEM 图像 {arr.shape[0]}x{arr.shape[1]} px, 平均衬度 {mean_i:.1f}, "
        f"RMS 粗糙度 {roughness_rms:.2f}, 检测到 {n_regions} 个暗区, "
        f"平均面积 {mean_area:.1f} px²"
    )

    data = {
        "summary": summary,
        "measurements": {
            "image_shape": [int(arr.shape[0]), int(arr.shape[1])],
            "pixel_size_nm": pixel_size,
            "contrast_mean": mean_i,
            "contrast_std": std_i,
            "contrast_p5": float(p5),
            "contrast_p50": float(p50),
            "contrast_p95": float(p95),
            "surface_roughness_rms": roughness_rms,
            "edge_density": edge_density,
            "threshold_used": float(thr),
            "n_dark_regions": int(n_regions),
            "dark_region_areas_px2": [float(a) for a in areas[:200]],
            "mean_area_px2": mean_area,
            "mean_area_nm2": mean_area * pixel_size ** 2,
        },
        "histogram": hist.astype(float).tolist(),
    }
    return ToolResult(data=data)
