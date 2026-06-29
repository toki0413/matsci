"""EDS 元素叠加场景: overlay / rgb / side_by_side 三种模式."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ._mpl_utils import load_gray, load_rgb, parse_color, save_figure, setup_matplotlib

if TYPE_CHECKING:
    from .tool import ImageDesignInput

from huginn.types import ToolResult

logger = logging.getLogger(__name__)


def eds_overlay(args: ImageDesignInput) -> ToolResult:
    setup_matplotlib()
    import matplotlib.pyplot as plt

    base_path = args.parameters.get("base_path")
    if not base_path:
        return ToolResult(
            data=None, success=False, error="eds_overlay 需要 parameters.base_path"
        )
    if not Path(base_path).exists():
        return ToolResult(
            data=None, success=False, error=f"基图不存在: {base_path}"
        )

    base = load_rgb(base_path)
    H, W, _ = base.shape
    element_maps = args.parameters.get("element_maps", [])
    mode = args.parameters.get("mode", "overlay")
    title = args.parameters.get("title", None)

    # 加载每个元素图, 算归一化强度 + 阈值 mask
    elem_data: list[dict[str, Any]] = []
    for em in element_maps:
        path = em.get("path", "")
        elem = em.get("element", "X")
        color = em.get("color", "#FF0000")
        threshold = float(em.get("threshold", 0.3))
        if not Path(path).exists():
            logger.debug("元素图不存在, 跳过: %s", path)
            continue
        try:
            g = load_gray(path)
            g_norm = (g - g.min()) / max(g.max() - g.min(), 1e-6)
            mask = g_norm > threshold
            elem_data.append(
                {
                    "element": elem,
                    "color": color,
                    "mask": mask,
                    "intensity": g_norm,
                    "coverage": float(mask.sum() / max(mask.size, 1)),
                }
            )
        except Exception as exc:
            logger.debug("加载元素图 %s 失败: %s", path, exc)

    if mode == "rgb" and len(elem_data) >= 3:
        # 三元素分别打到 R/G/B 通道
        fig, ax = plt.subplots(figsize=(10, 10))
        rgb_img = np.zeros((H, W, 3), dtype=float)
        for i, ed in enumerate(elem_data[:3]):
            c = parse_color(ed["color"]) / 255.0
            rgb_img[:, :, 0] += ed["intensity"] * c[0]
            rgb_img[:, :, 1] += ed["intensity"] * c[1]
            rgb_img[:, :, 2] += ed["intensity"] * c[2]
        rgb_img = np.clip(rgb_img, 0, 1)
        ax.imshow(rgb_img)
        legend_text = " + ".join([ed["element"] for ed in elem_data[:3]])
        ax.text(
            0.02,
            0.98,
            legend_text,
            transform=ax.transAxes,
            color="white",
            fontweight="bold",
            fontsize=18,
            va="top",
            bbox=dict(facecolor="black", alpha=0.6),
        )
        ax.set_axis_off()
    elif mode == "side_by_side":
        n = len(elem_data) + 1
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
        if n == 1:
            axes = [axes]
        axes[0].imshow(base.astype(np.uint8))
        axes[0].set_title("Base", fontweight="bold")
        axes[0].set_axis_off()
        for i, ed in enumerate(elem_data):
            axes[i + 1].imshow(ed["intensity"], cmap="hot")
            axes[i + 1].set_title(ed["element"], fontweight="bold")
            axes[i + 1].set_axis_off()
    else:
        # overlay: 基图 + 半透明伪彩
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(base.astype(np.uint8))
        for idx, ed in enumerate(elem_data):
            c = parse_color(ed["color"])
            overlay = np.zeros((H, W, 4), dtype=float)
            mask = ed["mask"]
            overlay[mask, 0] = c[0] / 255.0
            overlay[mask, 1] = c[1] / 255.0
            overlay[mask, 2] = c[2] / 255.0
            overlay[mask, 3] = 0.6
            ax.imshow(overlay)
            ax.text(
                0.02,
                0.98 - 0.06 * idx,
                ed["element"],
                transform=ax.transAxes,
                color=ed["color"],
                fontweight="bold",
                fontsize=18,
                va="top",
                bbox=dict(facecolor="black", alpha=0.5),
            )
        ax.set_axis_off()

    if title:
        fig.suptitle(title, fontweight="bold")

    save_figure(fig, args.output_path)

    coverage_str = ", ".join(
        f"{ed['element']}: {ed['coverage'] * 100:.1f}%" for ed in elem_data
    )
    summary = f"EDS 叠加 ({mode}): {len(elem_data)} 个元素, {coverage_str}"
    metadata = {
        "action": "eds_overlay",
        "mode": mode,
        "image_shape": [int(H), int(W)],
        "elements": [
            {
                "element": ed["element"],
                "color": ed["color"],
                "coverage_fraction": ed["coverage"],
            }
            for ed in elem_data
        ],
    }
    return ToolResult(
        data={
            "output_path": args.output_path,
            "summary": summary,
            "metadata": metadata,
        }
    )
