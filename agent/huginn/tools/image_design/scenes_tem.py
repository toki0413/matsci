"""TEM FFT 标注场景: FFT 频谱 + 自动亮斑检测 + d-spacing 标注."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ._mpl_utils import load_gray, save_figure, setup_matplotlib

if TYPE_CHECKING:
    from .tool import ImageDesignInput

from huginn.types import ToolResult

logger = logging.getLogger(__name__)


def tem_fft_annotate(args: ImageDesignInput) -> ToolResult:
    setup_matplotlib()
    import matplotlib.pyplot as plt

    input_path = args.parameters.get("input_path")
    if not input_path:
        return ToolResult(
            data=None,
            success=False,
            error="tem_fft_annotate 需要 parameters.input_path",
        )
    if not Path(input_path).exists():
        return ToolResult(
            data=None, success=False, error=f"输入图片不存在: {input_path}"
        )

    arr = load_gray(input_path)
    H, W = arr.shape
    do_fft = bool(args.parameters.get("fft", True))
    spots = args.parameters.get("spots", None)
    pixel_size_nm = float(args.parameters.get("pixel_size_nm", 0.01))
    auto_detect = bool(args.parameters.get("auto_detect", False))
    title = args.parameters.get("title", None)

    fft_img: np.ndarray | None = None
    if spots is None:
        spots = []

    if do_fft:
        try:
            from scipy.fft import fft2, fftshift
            from scipy.ndimage import maximum_filter

            fft = fftshift(fft2(arr - arr.mean()))
            power = np.abs(fft) ** 2
            fft_img = np.log1p(power)
            # 归一化到 0-1 便于显示
            pmin, pmax = float(fft_img.min()), float(fft_img.max())
            fft_img = (fft_img - pmin) / max(pmax - pmin, 1e-6)

            # 自动检测亮斑
            if auto_detect:
                cy, cx = np.array(power.shape) // 2
                y_idx, x_idx = np.indices(power.shape)
                r = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)
                power_masked = power.copy()
                power_masked[r < min(power.shape) * 0.05] = 0
                local_max = maximum_filter(power_masked, size=15)
                peaks = (
                    (power_masked == local_max)
                    & (power_masked > power_masked.max() * 0.1)
                )
                ys, xs = np.where(peaks)
                if len(ys) > 0:
                    vals = power[ys, xs]
                    order = np.argsort(vals)[::-1][:10]
                    N = float(max(arr.shape))
                    for i in order:
                        r_val = float(
                            np.sqrt((xs[i] - cx) ** 2 + (ys[i] - cy) ** 2)
                        )
                        if r_val <= 0:
                            continue
                        d_nm = pixel_size_nm * N / r_val
                        spots.append(
                            {
                                "qx": int(xs[i] - cx),
                                "qy": int(ys[i] - cy),
                                "d_spacing": float(d_nm),
                                "hkl": "",
                                "color": "#FF5722",
                            }
                        )
        except ImportError as exc:
            logger.debug("scipy 不可用, 跳过 FFT: %s", exc)

    # 整理 spot 信息, 补 d-spacing
    d_entries: list[dict[str, Any]] = []
    cy, cx = np.array(arr.shape) // 2
    N = float(max(arr.shape))
    for sp in spots:
        d_val = sp.get("d_spacing")
        if d_val is None:
            qx = float(sp.get("qx", 0))
            qy = float(sp.get("qy", 0))
            r_val = float(np.sqrt(qx ** 2 + qy ** 2))
            d_nm = pixel_size_nm * N / r_val if r_val > 0 else 0.0
        else:
            d_nm = float(d_val)
        d_entries.append(
            {
                "d_spacing_nm": float(d_nm),
                "hkl": sp.get("hkl", ""),
                "color": sp.get("color", "#FF5722"),
                "qx": sp.get("qx"),
                "qy": sp.get("qy"),
            }
        )

    # 左原图, 右 FFT
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(arr, cmap="gray")
    axes[0].set_title("TEM Image", fontweight="bold")
    axes[0].set_axis_off()

    if fft_img is not None:
        axes[1].imshow(fft_img, cmap="magma")
        axes[1].set_title("FFT", fontweight="bold")
        for sp in spots:
            qx = float(sp.get("qx", 0))
            qy = float(sp.get("qy", 0))
            c = sp.get("color", "#FF5722")
            axes[1].scatter(
                [cx + qx],
                [cy + qy],
                c=c,
                s=80,
                edgecolors="white",
                linewidths=1.5,
                zorder=5,
            )
            d_label = sp.get("d_spacing")
            if d_label is not None:
                axes[1].text(
                    cx + qx + 5,
                    cy + qy,
                    f"d={float(d_label):.3f}nm",
                    color="white",
                    fontweight="bold",
                    fontsize=12,
                    bbox=dict(facecolor="black", alpha=0.6),
                )
    else:
        axes[1].text(
            0.5,
            0.5,
            "FFT N/A (scipy missing)",
            transform=axes[1].transAxes,
            ha="center",
            fontweight="bold",
        )
    axes[1].set_axis_off()

    if title:
        fig.suptitle(title, fontweight="bold")

    save_figure(fig, args.output_path)

    summary = (
        f"TEM FFT 标注: 原图 {H}x{W} px, "
        f"{len(d_entries)} 个衍射斑点, pixel_size={pixel_size_nm} nm"
    )
    metadata = {
        "action": "tem_fft_annotate",
        "image_shape": [int(H), int(W)],
        "pixel_size_nm": pixel_size_nm,
        "fft_computed": bool(fft_img is not None),
        "auto_detect": auto_detect,
        "spots": d_entries,
    }
    return ToolResult(
        data={
            "output_path": args.output_path,
            "summary": summary,
            "metadata": metadata,
        }
    )
