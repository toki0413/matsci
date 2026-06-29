"""能带结构场景: 逐带绘制 + 费米能级 + 带隙箭头 + 高对称点."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from ._mpl_utils import get_param, save_figure, setup_matplotlib

if TYPE_CHECKING:
    from .tool import ImageDesignInput

from huginn.types import ToolResult

logger = logging.getLogger(__name__)


def band_diagram(args: ImageDesignInput) -> ToolResult:
    setup_matplotlib()
    import matplotlib.pyplot as plt

    kpoints = np.array(get_param(args, "kpoints", []), dtype=float)
    bands = get_param(args, "bands", [])
    if kpoints.size == 0 or not bands:
        return ToolResult(
            data=None, success=False, error="kpoints / bands 不能为空"
        )

    fermi_level = float(get_param(args, "fermi_level", 0.0))
    band_gap = get_param(args, "band_gap", None)
    k_labels = get_param(args, "k_labels", None)
    k_positions = get_param(args, "k_positions", None)
    title = get_param(args, "title", "Band Structure")

    fig, ax = plt.subplots(figsize=(10, 7))

    # 逐条画带
    for i, band in enumerate(bands):
        energies = np.array(band.get("energies", []), dtype=float)
        if energies.size == 0:
            continue
        c = band.get("color", "#2196F3")
        lbl = band.get("label", f"Band {i + 1}")
        ax.plot(kpoints, energies, color=c, linewidth=2.2, label=lbl)

    # 费米能级虚线
    ax.axhline(
        fermi_level,
        color="#000000",
        linestyle="--",
        linewidth=2,
        alpha=0.8,
        label=f"E_F = {fermi_level}",
    )

    # 带隙箭头: 找 VBM/CBM
    if band_gap is not None and float(band_gap) > 0:
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
        if below and above:
            vbm = float(max(b.max() for b in below))
            cbm = float(min(b.min() for b in above))
            mid_k = float((kpoints.min() + kpoints.max()) / 2)
            ax.annotate(
                "",
                xy=(mid_k, cbm),
                xytext=(mid_k, vbm),
                arrowprops=dict(
                    color="#F44336",
                    arrowstyle="<->",
                    linewidth=2.5,
                ),
            )
            ax.text(
                mid_k + 0.02 * (kpoints.max() - kpoints.min()),
                (vbm + cbm) / 2,
                f"E_g = {band_gap} eV",
                color="#F44336",
                fontweight="bold",
                fontsize=16,
            )

    # 高对称点竖线 + 标签
    if k_positions and k_labels:
        for kp in k_positions:
            ax.axvline(
                float(kp), color="#888888", linestyle=":", linewidth=1.2, alpha=0.6
            )
        ax.set_xticks([float(kp) for kp in k_positions])
        ax.set_xticklabels(list(k_labels))

    ax.set_xlabel("k-path", fontweight="bold")
    ax.set_ylabel("Energy (eV)", fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="best", framealpha=0.9)

    save_figure(fig, args.output_path)

    summary = f"能带结构: {len(bands)} 条带, E_F={fermi_level} eV"
    if band_gap is not None:
        summary += f", E_g={band_gap} eV"

    metadata = {
        "action": "band_diagram",
        "n_bands": int(len(bands)),
        "fermi_level_eV": fermi_level,
        "band_gap_eV": float(band_gap) if band_gap is not None else None,
        "k_labels": list(k_labels) if k_labels else None,
        "k_positions": [float(k) for k in (k_positions or [])],
    }
    return ToolResult(
        data={
            "output_path": args.output_path,
            "summary": summary,
            "metadata": metadata,
        }
    )
