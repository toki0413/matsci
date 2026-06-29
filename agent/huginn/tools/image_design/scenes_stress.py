"""应力-应变曲线场景: 弹性区虚线 + 屈服/极限/断裂三点标注."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from ._mpl_utils import get_param, save_figure, setup_matplotlib

if TYPE_CHECKING:
    from .tool import ImageDesignInput

from huginn.types import ToolResult

logger = logging.getLogger(__name__)


def stress_strain(args: ImageDesignInput) -> ToolResult:
    setup_matplotlib()
    import matplotlib.pyplot as plt

    strain = np.array(get_param(args, "strain", []), dtype=float)
    stress = np.array(get_param(args, "stress", []), dtype=float)
    if strain.size == 0 or stress.size == 0:
        return ToolResult(
            data=None, success=False, error="strain / stress 不能为空"
        )

    yield_point = get_param(args, "yield_point", None)
    ultimate_point = get_param(args, "ultimate_point", None)
    fracture_point = get_param(args, "fracture_point", None)
    youngs_modulus = get_param(args, "youngs_modulus", None)
    title = get_param(args, "title", "Stress-Strain Curve")
    color = get_param(args, "color", "#4CAF50")

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(strain, stress, color=color, linewidth=2.5, label="Curve")

    # 弹性区虚线
    if youngs_modulus is not None and strain.size > 1:
        eps_elastic_max = 0.005
        if yield_point:
            eps_elastic_max = float(yield_point[0])
        eps_line = np.linspace(0, eps_elastic_max, 50)
        # E 给的是 GPa, stress 是 MPa
        stress_line = eps_line * float(youngs_modulus) * 1000.0
        ax.plot(
            eps_line,
            stress_line,
            color="#FF5722",
            linestyle="--",
            linewidth=2,
            label=f"E = {youngs_modulus} GPa",
        )

    # 屈服 / 极限 / 断裂三点
    annotations: list[dict[str, Any]] = []
    x_range = float(strain.max() - strain.min())
    for pt, lbl, c in [
        (yield_point, "Yield", "#2196F3"),
        (ultimate_point, "UTS", "#F44336"),
        (fracture_point, "Fracture", "#9C27B0"),
    ]:
        if pt is None:
            continue
        ex, sx = float(pt[0]), float(pt[1])
        ax.scatter(
            [ex],
            [sx],
            color=c,
            s=120,
            zorder=5,
            edgecolor="black",
            linewidth=1.5,
        )
        ax.annotate(
            f"{lbl}\n({ex:.4f}, {sx:.1f} MPa)",
            xy=(ex, sx),
            xytext=(ex + 0.02 * x_range, sx),
            color=c,
            fontweight="bold",
            fontsize=14,
            arrowprops=dict(color=c, arrowstyle="->", linewidth=1.5),
        )
        annotations.append({"label": lbl, "strain": ex, "stress_MPa": sx})

    ax.set_xlabel("Strain", fontweight="bold")
    ax.set_ylabel("Stress (MPa)", fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    save_figure(fig, args.output_path)

    summary = f"应力-应变曲线: {strain.size} 点"
    if youngs_modulus is not None:
        summary += f", E={youngs_modulus} GPa"
    if ultimate_point:
        summary += f", UTS={ultimate_point[1]:.1f} MPa"

    metadata = {
        "action": "stress_strain",
        "n_points": int(strain.size),
        "youngs_modulus_GPa": (
            float(youngs_modulus) if youngs_modulus is not None else None
        ),
        "yield_point": list(yield_point) if yield_point else None,
        "ultimate_point": list(ultimate_point) if ultimate_point else None,
        "fracture_point": list(fracture_point) if fracture_point else None,
        "annotations": annotations,
    }
    return ToolResult(
        data={
            "output_path": args.output_path,
            "summary": summary,
            "metadata": metadata,
        }
    )
