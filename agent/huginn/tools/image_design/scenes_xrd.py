"""XRD 谱图场景: 多相叠加 + 峰位竖线 + hkl 标注 + Bragg d-spacing."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from ._mpl_utils import get_param, save_figure, setup_matplotlib

if TYPE_CHECKING:
    from .tool import ImageDesignInput

from huginn.types import ToolResult

logger = logging.getLogger(__name__)


def xrd_pattern(args: ImageDesignInput) -> ToolResult:
    setup_matplotlib()
    import matplotlib.pyplot as plt

    two_theta = np.array(get_param(args, "two_theta", []), dtype=float)
    intensity = np.array(get_param(args, "intensity", []), dtype=float)
    if two_theta.size == 0 or intensity.size == 0:
        return ToolResult(
            data=None, success=False, error="two_theta / intensity 不能为空"
        )

    peaks = get_param(args, "peaks", None)
    phases = get_param(args, "phases", None)
    title = get_param(args, "title", "XRD Pattern")
    wavelength = float(get_param(args, "wavelength", 1.5406))

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(
        two_theta, intensity, color="#2196F3", linewidth=1.8, label="Measured"
    )

    # 多相叠加 (虚线)
    if phases:
        for ph in phases:
            tt = np.array(ph.get("two_theta", []), dtype=float)
            ii = np.array(ph.get("intensity", []), dtype=float)
            if tt.size == 0:
                continue
            c = ph.get("color", "#FF5722")
            lbl = ph.get("label", "phase")
            ax.plot(
                tt,
                ii,
                color=c,
                linewidth=1.5,
                linestyle="--",
                alpha=0.8,
                label=lbl,
            )

    # 峰位竖线 + hkl 标注, 顺便算 d-spacing (Bragg)
    peak_info: list[dict[str, Any]] = []
    y_top = float(intensity.max()) if intensity.size else 1.0
    if peaks:
        for pk in peaks:
            tt_pk = float(pk.get("two_theta", 0))
            hkl = pk.get("hkl", "")
            phase_name = pk.get("phase", "")
            theta_rad = np.deg2rad(tt_pk / 2.0)
            sin_t = np.sin(theta_rad)
            d_nm = (
                wavelength / (2.0 * sin_t) * 0.1 if sin_t > 1e-6 else 0.0
            )  # Å → nm
            ax.axvline(
                tt_pk, color="#F44336", linestyle=":", linewidth=1.5, alpha=0.7
            )
            label_text = f"{phase_name} {hkl}".strip()
            ax.text(
                tt_pk,
                y_top * 0.95,
                f" {label_text}\n d={d_nm:.3f}nm",
                color="#F44336",
                fontweight="bold",
                rotation=90,
                va="top",
                fontsize=14,
            )
            peak_info.append(
                {
                    "two_theta": tt_pk,
                    "hkl": hkl,
                    "phase": phase_name,
                    "d_spacing_nm": float(d_nm),
                }
            )

    ax.set_xlabel("2θ (degree)", fontweight="bold")
    ax.set_ylabel("Intensity (a.u.)", fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xlim(float(two_theta.min()), float(two_theta.max()))

    save_figure(fig, args.output_path)

    summary = (
        f"XRD 谱图: {two_theta.size} 个数据点, "
        f"标注 {len(peaks or [])} 个峰, 波长 {wavelength} Å"
    )
    metadata = {
        "action": "xrd_pattern",
        "n_points": int(two_theta.size),
        "n_peaks": int(len(peaks or [])),
        "wavelength_A": wavelength,
        "peaks": peak_info,
        "n_phases": int(len(phases or [])),
    }
    return ToolResult(
        data={
            "output_path": args.output_path,
            "summary": summary,
            "metadata": metadata,
        }
    )
