"""粒度分布场景: 直方图 + lognormal 拟合 + 累计曲线 + D10/D50/D90."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from ._mpl_utils import get_param, save_figure, setup_matplotlib

if TYPE_CHECKING:
    from .tool import ImageDesignInput

from huginn.types import ToolResult

logger = logging.getLogger(__name__)


def particle_distribution(args: ImageDesignInput) -> ToolResult:
    setup_matplotlib()
    import matplotlib.pyplot as plt

    sizes = np.array(get_param(args, "sizes", []), dtype=float)
    sizes = sizes[sizes > 0]
    if sizes.size == 0:
        return ToolResult(
            data=None, success=False, error="sizes 为空或全为非正数"
        )

    bins = int(get_param(args, "bins", 20))
    title = get_param(args, "title", "Particle Size Distribution")
    x_label = get_param(args, "x_label", "Particle Size (nm)")
    y_label = get_param(args, "y_label", "Frequency")
    color = get_param(args, "color", "#2196F3")
    show_fit = bool(get_param(args, "show_fit", True))
    show_cumulative = bool(get_param(args, "show_cumulative", True))

    d10 = float(np.percentile(sizes, 10))
    d50 = float(np.percentile(sizes, 50))
    d90 = float(np.percentile(sizes, 90))
    mean_d = float(sizes.mean())
    std_d = float(sizes.std())

    fig, ax = plt.subplots(figsize=(10, 7))
    n_counts, bin_edges, _ = ax.hist(
        sizes,
        bins=bins,
        color=color,
        alpha=0.65,
        edgecolor="black",
        linewidth=0.8,
        label="Histogram",
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = float(bin_edges[1] - bin_edges[0])

    # lognormal 拟合叠加
    fit_mu = None
    fit_sigma = None
    if show_fit and sizes.size >= 5:
        try:
            from scipy.stats import lognorm

            shape, loc, scale = lognorm.fit(sizes, floc=0.0)
            fit_sigma = float(shape)
            fit_mu = float(np.log(scale))
            x_fit = np.linspace(sizes.min(), sizes.max(), 200)
            pdf = (
                lognorm.pdf(x_fit, shape, loc=0.0, scale=scale)
                * sizes.size
                * bin_width
            )
            ax.plot(
                x_fit,
                pdf,
                color="#FF5722",
                linewidth=2.5,
                label=f"Lognormal fit (μ={fit_mu:.3f}, σ={fit_sigma:.3f})",
            )
        except Exception as exc:
            logger.debug("lognormal 拟合失败: %s", exc)

    # 累计曲线放右轴
    if show_cumulative:
        ax2 = ax.twinx()
        cum = np.cumsum(n_counts) / max(n_counts.sum(), 1) * 100.0
        ax2.plot(
            bin_centers,
            cum,
            color="#4CAF50",
            linewidth=2.5,
            marker="o",
            markersize=5,
            label="Cumulative %",
        )
        ax2.set_ylabel("Cumulative (%)", fontweight="bold")
        ax2.set_ylim(0, 105)

    # D10/D50/D90 竖线
    y_top = ax.get_ylim()[1]
    for val, lbl, c in [
        (d10, "D10", "#9C27B0"),
        (d50, "D50", "#F44336"),
        (d90, "D90", "#FF9800"),
    ]:
        ax.axvline(val, color=c, linestyle="--", linewidth=2, alpha=0.8)
        ax.text(
            val,
            y_top * 0.95,
            f" {lbl}={val:.2f}",
            color=c,
            fontweight="bold",
            rotation=90,
            va="top",
        )

    ax.set_xlabel(x_label, fontweight="bold")
    ax.set_ylabel(y_label, fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.9)
    if show_cumulative:
        ax2.legend(loc="center right", framealpha=0.9)

    save_figure(fig, args.output_path)

    summary = (
        f"粒度分布: N={sizes.size}, D10={d10:.3f} nm, D50={d50:.3f} nm, "
        f"D90={d90:.3f} nm, mean={mean_d:.3f} ± {std_d:.3f} nm"
    )
    metadata = {
        "action": "particle_distribution",
        "n_particles": int(sizes.size),
        "d10_nm": d10,
        "d50_nm": d50,
        "d90_nm": d90,
        "mean_nm": mean_d,
        "std_nm": std_d,
        "lognormal_mu": fit_mu,
        "lognormal_sigma": fit_sigma,
        "bins": bins,
    }
    return ToolResult(
        data={
            "output_path": args.output_path,
            "summary": summary,
            "metadata": metadata,
        }
    )
