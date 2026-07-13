"""相场后处理 — 灰度量化分相 + 体积分数 + 界面宽度 + domain 形态学."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from huginn.types import ToolResult
from huginn.tools.image_analysis._utils import load_gray

if TYPE_CHECKING:
    from huginn.tools.image_analysis.tool import ImageAnalysisInput

logger = logging.getLogger(__name__)


def phase_field(args: "ImageAnalysisInput") -> ToolResult:
    arr = load_gray(args.image_path)
    n_phases = int(args.parameters.get("n_phases", 2))
    interface_width_px = args.parameters.get("interface_width_px", None)
    pixel_size = float(args.parameters.get("pixel_size_nm", 1.0))

    n_phases = max(2, min(n_phases, 10))

    # 等间距量化把灰度分成 n_phases 个相
    a_min = float(arr.min())
    a_max = float(arr.max()) + 1e-6
    bins = np.linspace(a_min, a_max, n_phases + 1)
    phase_map = np.digitize(arr, bins) - 1
    phase_map = np.clip(phase_map, 0, n_phases - 1)

    # 体积分数
    total = int(phase_map.size)
    vol_fracs = {
        f"phase_{i + 1}": float((phase_map == i).sum() / total)
        for i in range(n_phases)
    }

    # 界面像素: 相邻像素属于不同相
    diff_x = np.abs(np.diff(phase_map, axis=1))
    diff_y = np.abs(np.diff(phase_map, axis=0))
    interface_mask = np.zeros_like(phase_map, dtype=bool)
    interface_mask[:, :-1] |= diff_x > 0
    interface_mask[:, 1:] |= diff_x > 0
    interface_mask[:-1, :] |= diff_y > 0
    interface_mask[1:, :] |= diff_y > 0

    interface_pixels = int(interface_mask.sum())
    interface_fraction = float(interface_pixels / total)

    # 界面宽度估计
    if interface_width_px is None:
        try:
            from scipy.ndimage import sobel

            grad = np.sqrt(sobel(arr, axis=0) ** 2 + sobel(arr, axis=1) ** 2)
            if interface_pixels > 0:
                grad_in = grad[interface_mask]
                grad_mean = float(grad_in.mean())
            else:
                grad_mean = 0.0
            contrast_range = a_max - a_min + 1e-6
            # 粗估: 阶跃宽度 ≈ contrast_range / grad_mean
            interface_width_px = float(
                contrast_range / max(grad_mean, 1e-6)
            )
        except ImportError:
            interface_width_px = 1.0
    else:
        interface_width_px = float(interface_width_px)

    # 每个相的 domain 形态学
    phase_morph: dict[str, Any] = {}
    try:
        from scipy.ndimage import label as nd_label, center_of_mass

        for i in range(n_phases):
            mask = phase_map == i
            labeled, n_dom = nd_label(mask)
            counts = np.bincount(labeled.ravel())
            counts[0] = 0
            domain_areas = counts[counts > 0]
            if len(domain_areas) > 0:
                ecd = np.sqrt(4.0 * domain_areas / np.pi)
                # top-3 domain 质心 — 让 visual primitives 能指向最大 domain 位置
                valid_labels = np.where(counts > 0)[0]
                sorted_labels = valid_labels[np.argsort(-counts[valid_labels])]
                top_centroids: list[list[float]] = []
                for lbl in sorted_labels[:3]:
                    com = center_of_mass(mask, labeled, lbl)
                    top_centroids.append([float(com[1]), float(com[0])])  # [x, y]
                phase_morph[f"phase_{i + 1}"] = {
                    "n_domains": int(n_dom),
                    "mean_domain_area_px2": float(domain_areas.mean()),
                    "max_domain_area_px2": float(domain_areas.max()),
                    "mean_domain_ecd_px": float(ecd.mean()),
                    "top_domain_centroids_px": top_centroids,
                }
            else:
                phase_morph[f"phase_{i + 1}"] = {
                    "n_domains": 0,
                    "mean_domain_area_px2": 0.0,
                    "max_domain_area_px2": 0.0,
                    "mean_domain_ecd_px": 0.0,
                    "top_domain_centroids_px": [],
                }
    except ImportError:
        phase_morph = {
            f"phase_{i + 1}": {"note": "scipy.ndimage 不可用, 跳过 domain 形态学"}
            for i in range(n_phases)
        }

    vol_str = ", ".join(f"{k}: {v * 100:.1f}%" for k, v in vol_fracs.items())
    summary = (
        f"相场分析: {n_phases} 个相, {vol_str}, "
        f"界面占比 {interface_fraction * 100:.2f}%"
    )

    data = {
        "summary": summary,
        "measurements": {
            "image_shape": [int(arr.shape[0]), int(arr.shape[1])],
            "n_phases": n_phases,
            "phase_thresholds": [float(b) for b in bins],
            "volume_fractions": vol_fracs,
            "interface_pixel_fraction": interface_fraction,
            "interface_width_px": float(interface_width_px),
            "interface_width_nm": float(interface_width_px * pixel_size),
            "morphology": phase_morph,
        },
    }
    return ToolResult(data=data)
