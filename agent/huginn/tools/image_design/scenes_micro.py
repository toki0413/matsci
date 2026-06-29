"""显微结构标注场景: 圆/矩形/箭头/文字标注 + 比例尺."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ._mpl_utils import load_rgb, save_figure, setup_matplotlib

if TYPE_CHECKING:
    from .tool import ImageDesignInput

from huginn.types import ToolResult

logger = logging.getLogger(__name__)


def microstructure_annotate(args: ImageDesignInput) -> ToolResult:
    setup_matplotlib()
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    input_path = args.parameters.get("input_path")
    if not input_path:
        return ToolResult(
            data=None,
            success=False,
            error="microstructure_annotate 需要 parameters.input_path",
        )
    if not Path(input_path).exists():
        return ToolResult(
            data=None, success=False, error=f"输入图片不存在: {input_path}"
        )

    rgb = load_rgb(input_path)
    H, W, _ = rgb.shape
    annotations = args.parameters.get("annotations", [])
    scale_bar = args.parameters.get("scale_bar", None)
    title = args.parameters.get("title", None)
    alpha = float(args.parameters.get("alpha", 0.8))

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb.astype(np.uint8))

    n_annot = 0
    for ann in annotations:
        t = ann.get("type", "text")
        c = ann.get("color", "#F44336")
        label = ann.get("label", "")
        try:
            if t == "circle":
                circ = mpatches.Circle(
                    (float(ann["x"]), float(ann["y"])),
                    float(ann["r"]),
                    fill=False,
                    edgecolor=c,
                    linewidth=2.5,
                    alpha=alpha,
                )
                ax.add_patch(circ)
                if label:
                    ax.text(
                        float(ann["x"]) + float(ann["r"]),
                        float(ann["y"]),
                        label,
                        color=c,
                        fontweight="bold",
                        fontsize=14,
                    )
                n_annot += 1
            elif t == "rect":
                rect = mpatches.Rectangle(
                    (float(ann["x"]), float(ann["y"])),
                    float(ann["w"]),
                    float(ann["h"]),
                    fill=False,
                    edgecolor=c,
                    linewidth=2.5,
                    alpha=alpha,
                )
                ax.add_patch(rect)
                if label:
                    ax.text(
                        float(ann["x"]),
                        float(ann["y"]) - 5,
                        label,
                        color=c,
                        fontweight="bold",
                        fontsize=14,
                    )
                n_annot += 1
            elif t == "arrow":
                dx = float(ann.get("dx", 0))
                dy = float(ann.get("dy", 0))
                ax.annotate(
                    "",
                    xy=(float(ann["x"]) + dx, float(ann["y"]) + dy),
                    xytext=(float(ann["x"]), float(ann["y"])),
                    arrowprops=dict(
                        color=c,
                        arrowstyle="->",
                        linewidth=2.5,
                        alpha=alpha,
                    ),
                )
                if label:
                    ax.text(
                        float(ann["x"]) + dx,
                        float(ann["y"]) + dy,
                        label,
                        color=c,
                        fontweight="bold",
                        fontsize=14,
                    )
                n_annot += 1
            elif t == "text":
                txt = ann.get("text", "")
                ax.text(
                    float(ann["x"]),
                    float(ann["y"]),
                    txt,
                    color=c,
                    fontweight="bold",
                    fontsize=16,
                    alpha=alpha,
                )
                n_annot += 1
        except (KeyError, TypeError, ValueError) as exc:
            logger.debug("跳过标注 %s: %s", ann, exc)

    # 比例尺
    scale_info = None
    if scale_bar:
        length_nm = float(scale_bar.get("length_nm", 100))
        position = scale_bar.get("position", "bottom_right")
        pixel_size_nm = float(scale_bar.get("pixel_size_nm", 1.0))
        bar_len_px = max(10, int(length_nm / max(pixel_size_nm, 1e-6)))
        if position == "bottom_right":
            bx = W - bar_len_px - 20
            by = H - 20
        elif position == "bottom_left":
            bx = 20
            by = H - 20
        elif position == "top_right":
            bx = W - bar_len_px - 20
            by = 30
        else:
            bx = 20
            by = 30
        # 黑底白线, 双描边保证任何背景都看得见
        ax.plot([bx, bx + bar_len_px], [by, by], color="black", linewidth=6)
        ax.plot([bx, bx + bar_len_px], [by, by], color="white", linewidth=3)
        ax.text(
            bx + bar_len_px / 2,
            by - 15,
            f"{length_nm} nm",
            color="white",
            ha="center",
            fontweight="bold",
            fontsize=14,
            bbox=dict(facecolor="black", alpha=0.6, edgecolor="none"),
        )
        scale_info = {
            "length_nm": length_nm,
            "pixel_size_nm": pixel_size_nm,
            "bar_length_px": bar_len_px,
            "position": position,
        }

    ax.set_axis_off()
    if title:
        ax.set_title(title, fontweight="bold")

    save_figure(fig, args.output_path)

    summary = (
        f"显微图标注: 原图 {H}x{W} px, {n_annot} 个标注"
        + (", 带比例尺" if scale_bar else "")
    )
    metadata = {
        "action": "microstructure_annotate",
        "image_shape": [int(H), int(W)],
        "n_annotations": int(n_annot),
        "scale_bar": scale_info,
    }
    return ToolResult(
        data={
            "output_path": args.output_path,
            "summary": summary,
            "metadata": metadata,
        }
    )
