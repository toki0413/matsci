"""Visualization layer for the unified scientific computing framework."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from huginn.unified.core import UnifiedProblem
from huginn.unified.solve import solve


def plot_solution(
    mesh: list[float] | list[list[float]],
    solution: list[float],
    output_path: str | Path,
    title: str = "Unified solution",
    xlabel: str = "x",
    ylabel: str = "y",
    shape: tuple[int, int] | None = None,
) -> Path:
    """Plot a 1D or 2D solution field and save to file.

    Args:
        mesh: Node coordinates (1D list or 2D [[x,y], ...]).
        solution: Field values at nodes.
        output_path: Path to save the figure.
        title: Plot title.
        xlabel, ylabel: Axis labels.
        shape: For 2D regular grids, the (nx, ny) shape.

    Returns:
        Path to the saved figure.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))

    if shape is not None and len(shape) == 2:
        # 2D regular grid heatmap
        nx, ny = shape
        u = np.array(solution).reshape(nx, ny)
        im = ax.imshow(
            u, origin="lower", extent=[0, 1, 0, 1], cmap="viridis", aspect="auto"
        )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.colorbar(im, ax=ax)
    else:
        # 1D line plot
        ax.plot(mesh, solution, marker="o", linestyle="-")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def solve_and_plot(
    problem: UnifiedProblem,
    method: str = "fem",
    n: int = 10,
    output_path: str | Path = "unified_solution.png",
) -> dict[str, Any]:
    """Solve a unified problem and save a plot of the solution."""
    sol = solve(problem, method=method, n=n)
    coords = problem.domain.coordinates if problem.domain else []
    xlabel = str(coords[0]) if coords else "x"
    ylabel = str(coords[1]) if len(coords) > 1 else "u"
    path = plot_solution(
        mesh=sol["mesh"],
        solution=sol["solution"],
        output_path=output_path,
        title=f"{problem.name} ({sol['method']}, n={n})",
        xlabel=xlabel,
        ylabel=ylabel,
        shape=tuple(sol["shape"]) if "shape" in sol else None,
    )
    return {
        "plot_path": str(path),
        **sol,
    }
