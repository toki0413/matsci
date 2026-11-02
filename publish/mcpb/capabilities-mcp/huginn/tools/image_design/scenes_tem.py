"""TEM FFT 标注场景: FFT 频谱 + 自动亮斑检测 + d-spacing 标注."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ._mpl_utils import load_gray, save_figure, setup_matplotlib

if TYPE_CHECKING:
    from .tool import ImageDesignInput

from huginn.core_types import ToolResult

logger = logging.getLogger(__name__)


@dataclass
class _TemParams:
    """Parsed tem_fft_annotate inputs."""

    arr: np.ndarray
    h: int
    w: int
    do_fft: bool
    spots: list[dict[str, Any]]
    pixel_size_nm: float
    auto_detect: bool
    title: str | None


def tem_fft_annotate(args: ImageDesignInput) -> ToolResult:
    setup_matplotlib()
    import matplotlib.pyplot as plt

    params = _parse_tem_params(args)
    if isinstance(params, ToolResult):
        return params

    fft_img: np.ndarray | None = None
    if params.do_fft:
        fft_img = _compute_fft(
            params.arr, params.pixel_size_nm, params.auto_detect, params.spots
        )

    d_entries = _build_d_entries(params.spots, params.arr.shape, params.pixel_size_nm)
    _render_tem_figure(plt, params, fft_img, args.output_path)

    summary = (
        f"TEM FFT 标注: 原图 {params.h}x{params.w} px, "
        f"{len(d_entries)} 个衍射斑点, pixel_size={params.pixel_size_nm} nm"
    )
    metadata = {
        "action": "tem_fft_annotate",
        "image_shape": [int(params.h), int(params.w)],
        "pixel_size_nm": params.pixel_size_nm,
        "fft_computed": bool(fft_img is not None),
        "auto_detect": params.auto_detect,
        "spots": d_entries,
    }
    return ToolResult(
        data={
            "output_path": args.output_path,
            "summary": summary,
            "metadata": metadata,
        }
    )


def _parse_tem_params(args: ImageDesignInput) -> _TemParams | ToolResult:
    """Validate the input image and normalize the remaining parameters."""
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
    h, w = arr.shape
    spots = args.parameters.get("spots", None)
    if spots is None:
        spots = []
    return _TemParams(
        arr=arr,
        h=h,
        w=w,
        do_fft=bool(args.parameters.get("fft", True)),
        spots=spots,
        pixel_size_nm=float(args.parameters.get("pixel_size_nm", 0.01)),
        auto_detect=bool(args.parameters.get("auto_detect", False)),
        title=args.parameters.get("title", None),
    )


def _compute_fft(
    arr: np.ndarray,
    pixel_size_nm: float,
    auto_detect: bool,
    spots: list[dict[str, Any]],
) -> np.ndarray | None:
    """FFT 频谱 + 归一化到 0-1; auto_detect 时追加衍射斑点."""
    try:
        from scipy.fft import fft2, fftshift
        from scipy.ndimage import maximum_filter

        fft = fftshift(fft2(arr - arr.mean()))
        power = np.abs(fft) ** 2
        fft_img = np.log1p(power)
        # 归一化到 0-1 便于显示
        pmin, pmax = float(fft_img.min()), float(fft_img.max())
        fft_img = (fft_img - pmin) / max(pmax - pmin, 1e-6)
    except ImportError as exc:
        logger.debug("scipy 不可用, 跳过 FFT: %s", exc)
        return None

    # 自动检测亮斑
    if auto_detect:
        _detect_fft_spots(power, pixel_size_nm, spots, maximum_filter)
    return fft_img


def _detect_fft_spots(
    power: np.ndarray,
    pixel_size_nm: float,
    spots: list[dict[str, Any]],
    maximum_filter: Any,
) -> None:
    """自动检测 FFT 亮斑, 按强度取前 10 个, 换算 d-spacing 后追加到 spots."""
    cy, cx = np.array(power.shape) // 2
    y_idx, x_idx = np.indices(power.shape)
    r = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)
    power_masked = power.copy()
    power_masked[r < min(power.shape) * 0.05] = 0
    local_max = maximum_filter(power_masked, size=15)
    peaks = (power_masked == local_max) & (power_masked > power_masked.max() * 0.1)
    ys, xs = np.where(peaks)
    if len(ys) == 0:
        return

    vals = power[ys, xs]
    order = np.argsort(vals)[::-1][:10]
    n_max = float(max(power.shape))
    for i in order:
        r_val = float(np.sqrt((xs[i] - cx) ** 2 + (ys[i] - cy) ** 2))
        if r_val <= 0:
            continue
        spots.append(
            {
                "qx": int(xs[i] - cx),
                "qy": int(ys[i] - cy),
                "d_spacing": float(pixel_size_nm * n_max / r_val),
                "hkl": "",
                "color": "#FF5722",
            }
        )


def _build_d_entries(
    spots: list[dict[str, Any]], shape: tuple[int, ...], pixel_size_nm: float
) -> list[dict[str, Any]]:
    """整理 spot 信息, 补 d-spacing."""
    n_max = float(max(shape))
    return [_spot_to_entry(sp, n_max, pixel_size_nm) for sp in spots]


def _spot_to_entry(
    sp: dict[str, Any], n_max: float, pixel_size_nm: float
) -> dict[str, Any]:
    """Normalize one spot into the reported d-spacing entry."""
    d_val = sp.get("d_spacing")
    if d_val is None:
        qx = float(sp.get("qx", 0))
        qy = float(sp.get("qy", 0))
        r_val = float(np.sqrt(qx**2 + qy**2))
        d_nm = pixel_size_nm * n_max / r_val if r_val > 0 else 0.0
    else:
        d_nm = float(d_val)
    return {
        "d_spacing_nm": float(d_nm),
        "hkl": sp.get("hkl", ""),
        "color": sp.get("color", "#FF5722"),
        "qx": sp.get("qx"),
        "qy": sp.get("qy"),
    }


def _render_tem_figure(
    plt: Any,
    params: _TemParams,
    fft_img: np.ndarray | None,
    output_path: str,
) -> None:
    """左原图, 右 FFT (含斑点标注)."""
    cy, cx = np.array(params.arr.shape) // 2
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(params.arr, cmap="gray")
    axes[0].set_title("TEM Image", fontweight="bold")
    axes[0].set_axis_off()

    if fft_img is not None:
        axes[1].imshow(fft_img, cmap="magma")
        axes[1].set_title("FFT", fontweight="bold")
        for sp in params.spots:
            _draw_spot_marker(axes[1], sp, cx, cy)
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

    if params.title:
        fig.suptitle(params.title, fontweight="bold")

    save_figure(fig, output_path)


def _draw_spot_marker(ax: Any, sp: dict[str, Any], cx: Any, cy: Any) -> None:
    """Draw one diffraction spot plus its d-spacing label on the FFT axes."""
    qx = float(sp.get("qx", 0))
    qy = float(sp.get("qy", 0))
    ax.scatter(
        [cx + qx],
        [cy + qy],
        c=sp.get("color", "#FF5722"),
        s=80,
        edgecolors="white",
        linewidths=1.5,
        zorder=5,
    )
    d_label = sp.get("d_spacing")
    if d_label is not None:
        ax.text(
            cx + qx + 5,
            cy + qy,
            f"d={float(d_label):.3f}nm",
            color="white",
            fontweight="bold",
            fontsize=12,
            bbox={"facecolor": "black", "alpha": 0.6},
        )


