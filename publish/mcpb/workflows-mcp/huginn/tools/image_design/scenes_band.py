"""能带结构场景: 逐带绘制 + 费米能级 + 带隙箭头 + 高对称点."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from ._mpl_utils import get_param, save_figure, setup_matplotlib

if TYPE_CHECKING:
    from .tool import ImageDesignInput

from huginn.core_types import ToolResult

logger = logging.getLogger(__name__)


def band_diagram(args: ImageDesignInput) -> ToolResult:
    setup_matplotlib()
    import matplotlib.pyplot as plt

    kpoints = np.array(get_param(args, "kpoints", []), dtype=float)
    bands = get_param(args, "bands", [])
    if kpoints.size == 0 or not bands:
        return ToolResult(data=None, success=False, error="kpoints / bands 不能为空")

    fermi_level = float(get_param(args, "fermi_level", 0.0))
    band_gap = get_param(args, "band_gap", None)
    k_labels = get_param(args, "k_labels", None)
    k_positions = get_param(args, "k_positions", None)
    title = get_param(args, "title", "Band Structure")

    fig, ax = plt.subplots(figsize=(10, 7))

    _plot_bands(ax, kpoints, bands)

    # 费米能级虚线
    ax.axhline(
        fermi_level,
        color="#000000",
        linestyle="--",
        linewidth=2,
        alpha=0.8,
        label=f"E_F = {fermi_level}",
    )

    _annotate_band_gap(ax, kpoints, bands, fermi_level, band_gap)
    _draw_high_symmetry_points(ax, k_positions, k_labels)

    ax.set_xlabel("k-path", fontweight="bold")
    ax.set_ylabel("Energy (eV)", fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="best", framealpha=0.9)

    save_figure(fig, args.output_path)

    summary = _band_summary(bands, fermi_level, band_gap)
    metadata = _band_metadata(bands, fermi_level, band_gap, k_labels, k_positions)
    return ToolResult(
        data={
            "output_path": args.output_path,
            "summary": summary,
            "metadata": metadata,
        }
    )


def _band_summary(bands, fermi_level, band_gap) -> str:
    """summary 文本, 带隙存在时追加 E_g."""
    summary = f"能带结构: {len(bands)} 条带, E_F={fermi_level} eV"
    if band_gap is not None:
        summary += f", E_g={band_gap} eV"
    return summary


def _band_metadata(bands, fermi_level, band_gap, k_labels, k_positions) -> dict:
    """metadata 字典."""
    return {
        "action": "band_diagram",
        "n_bands": int(len(bands)),
        "fermi_level_eV": fermi_level,
        "band_gap_eV": float(band_gap) if band_gap is not None else None,
        "k_labels": list(k_labels) if k_labels else None,
        "k_positions": [float(k) for k in (k_positions or [])],
    }


def _plot_bands(ax, kpoints, bands) -> None:
    """逐条画带, 空带跳过."""
    for i, band in enumerate(bands):
        energies = np.array(band.get("energies", []), dtype=float)
        if energies.size == 0:
            continue
        c = band.get("color", "#2196F3")
        lbl = band.get("label", f"Band {i + 1}")
        ax.plot(kpoints, energies, color=c, linewidth=2.2, label=lbl)


def _annotate_band_gap(ax, kpoints, bands, fermi_level, band_gap) -> None:
    """带隙箭头: 找 VBM/CBM."""
    if band_gap is None or not float(band_gap) > 0:
        return
    below = [
        np.array(b.get("energies", []), dtype=float)
        for b in bands
        if len(b.get("energies", [])) > 0
        and np.array(b["energies"], dtype=float).max() <= fermi_level
    ]
    above = [
        np.array(b.get("energies", []), dtype=float)
        for b in bands
        if len(b.get("energies", [])) > 0
        and np.array(b["energies"], dtype=float).min() >= fermi_level
    ]
    if not (below and above):
        return
    vbm = float(max(b.max() for b in below))
    cbm = float(min(b.min() for b in above))
    mid_k = float((kpoints.min() + kpoints.max()) / 2)
    ax.annotate(
        "",
        xy=(mid_k, cbm),
        xytext=(mid_k, vbm),
        arrowprops={
            "color": "#F44336",
            "arrowstyle": "<->",
            "linewidth": 2.5,
        },
    )
    ax.text(
        mid_k + 0.02 * (kpoints.max() - kpoints.min()),
        (vbm + cbm) / 2,
        f"E_g = {band_gap} eV",
        color="#F44336",
        fontweight="bold",
        fontsize=16,
    )


def _draw_high_symmetry_points(ax, k_positions, k_labels) -> None:
    """高对称点竖线 + 标签."""
    if not (k_positions and k_labels):
        return
    for kp in k_positions:
        ax.axvline(float(kp), color="#888888", linestyle=":", linewidth=1.2, alpha=0.6)
    ax.set_xticks([float(kp) for kp in k_positions])
    ax.set_xticklabels(list(k_labels))
