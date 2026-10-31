"""粒度统计 — Otsu 阈值 + 连通域 + D10/D50/D90 + 对数正态拟合."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from huginn.core_types import ToolResult
from huginn.tools.image_analysis._utils import load_gray, otsu_numpy

if TYPE_CHECKING:
    from huginn.tools.image_analysis.tool import ImageAnalysisInput

logger = logging.getLogger(__name__)


def _otsu_threshold(arr: np.ndarray) -> float:
    """Otsu 阈值, skimage 优先, 没有就走 numpy."""
    try:
        from skimage.filters import threshold_otsu

        return float(threshold_otsu(arr))
    except ImportError:
        return otsu_numpy(arr)


def _region_areas(binary: np.ndarray) -> np.ndarray | None:
    """连通域面积; skimage 优先, scipy 兜底, 都没装返回 None."""
    try:
        from skimage.measure import label as sk_label
        from skimage.measure import regionprops

        labeled = sk_label(binary)
        props = regionprops(labeled)
        return np.array([p.area for p in props], dtype=float)
    except ImportError:
        try:
            from scipy.ndimage import label as nd_label

            labeled, _ = nd_label(binary)
            counts = np.bincount(labeled.ravel())
            counts[0] = 0
            return counts[counts > 0].astype(float)
        except ImportError:
            return None


def _ecd_distribution(areas_f: np.ndarray, pixel_size: float, binning: str) -> dict:
    """等效圆直径 → 分位数 / 直方图 / 对数正态拟合."""
    # 等效圆直径
    ecd_px = np.sqrt(4.0 * areas_f / np.pi)
    ecd_nm = ecd_px * pixel_size

    d10 = float(np.percentile(ecd_nm, 10))
    d50 = float(np.percentile(ecd_nm, 50))
    d90 = float(np.percentile(ecd_nm, 90))

    # 直方图
    ecd_min = float(max(ecd_nm.min(), 1e-6))
    ecd_max = float(ecd_nm.max() + 1e-6)
    if binning == "log" and ecd_max > ecd_min:
        log_edges = np.linspace(np.log10(ecd_min), np.log10(ecd_max), 20)
        edges = 10**log_edges
    else:
        edges = np.linspace(ecd_min, ecd_max, 20)
    hist, _ = np.histogram(ecd_nm, bins=edges)

    # 对数正态拟合
    fit_mu: float | None = None
    fit_sigma: float | None = None
    try:
        from scipy.stats import lognorm

        shape, loc, scale = lognorm.fit(ecd_nm, floc=0.0)
        fit_sigma = float(shape)
        fit_mu = float(np.log(scale))
    except Exception:
        # 退化: 直接对 log 数据算 mean/std
        log_data = np.log(ecd_nm[ecd_nm > 0])
        if log_data.size > 0:
            fit_mu = float(log_data.mean())
            fit_sigma = float(log_data.std())

    return {
        "ecd_nm": ecd_nm,
        "d10": d10,
        "d50": d50,
        "d90": d90,
        "hist": hist,
        "fit_mu": fit_mu,
        "fit_sigma": fit_sigma,
    }


def _empty_data(otsu_t: float, invert: bool) -> dict:
    """未检测到颗粒时的 data."""
    return {
        "summary": "粒度统计: 未检测到符合面积范围的颗粒, 检查 min_area_px/max_area_px/invert",
        "measurements": {
            "n_particles": 0,
            "threshold_otsu": otsu_t,
            "invert": invert,
        },
    }


def _particles_data(
    arr: np.ndarray,
    pixel_size: float,
    otsu_t: float,
    invert: bool,
    min_area: int,
    max_area: int,
    areas_f: np.ndarray,
    dist: dict,
) -> dict:
    """组装 summary + measurements + histogram."""
    ecd_nm = dist["ecd_nm"]
    summary = (
        f"粒度统计: 检测到 {len(areas_f)} 个颗粒, "
        f"D50 = {dist['d50']:.3f} nm, D10/D90 = {dist['d10']:.3f}/{dist['d90']:.3f} nm"
    )
    return {
        "summary": summary,
        "measurements": {
            "image_shape": [int(arr.shape[0]), int(arr.shape[1])],
            "pixel_size_nm": pixel_size,
            "threshold_otsu": otsu_t,
            "invert": invert,
            "n_particles": int(len(areas_f)),
            "ecd_mean_nm": float(ecd_nm.mean()),
            "ecd_std_nm": float(ecd_nm.std()),
            "ecd_min_nm": float(ecd_nm.min()),
            "ecd_max_nm": float(ecd_nm.max()),
            "d10_nm": dist["d10"],
            "d50_nm": dist["d50"],
            "d90_nm": dist["d90"],
            "area_min_px": min_area,
            "area_max_px": max_area,
            "lognormal_mu": dist["fit_mu"],
            "lognormal_sigma": dist["fit_sigma"],
        },
        "histogram": dist["hist"].astype(float).tolist(),
    }


def particle_stats(args: ImageAnalysisInput) -> ToolResult:
    arr = load_gray(args.image_path)
    pixel_size = float(args.parameters.get("pixel_size_nm", 1.0))
    min_area = int(args.parameters.get("min_area_px", 10))
    max_area = int(
        args.parameters.get("max_area_px", (arr.shape[0] * arr.shape[1]) // 2)
    )
    binning = args.parameters.get("binning", "log")
    invert = bool(args.parameters.get("invert", False))

    # Otsu 阈值, skimage 优先, 没有就走 numpy
    otsu_t = _otsu_threshold(arr)

    # invert=True: 颗粒亮; 默认: 颗粒暗
    binary = arr > otsu_t if invert else arr < otsu_t

    # 连通域 + regionprops, skimage 优先, scipy 兜底
    areas = _region_areas(binary)
    if areas is None:
        return ToolResult(
            data=None,
            success=False,
            error="粒度统计需要 scipy.ndimage 或 skimage, 请安装其一",
        )

    # 面积过滤
    keep = (areas >= min_area) & (areas <= max_area)
    areas_f = areas[keep]

    if len(areas_f) == 0:
        return ToolResult(data=_empty_data(otsu_t, invert))

    dist = _ecd_distribution(areas_f, pixel_size, binning)
    data = _particles_data(
        arr, pixel_size, otsu_t, invert, min_area, max_area, areas_f, dist
    )
    return ToolResult(data=data)
