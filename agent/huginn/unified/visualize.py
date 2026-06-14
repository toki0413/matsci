"""Visualization layer for the unified scientific computing framework."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

from huginn.unified.core import UnifiedProblem
from huginn.unified.solve import solve


def plot_solution(
    mesh: list[float],
    solution: list[float],
    output_path: str | Path,
    title: str = "Unified solution",
    xlabel: str = "x",
    ylabel: str = "u",
) -> Path:
    """Plot a 1D solution field and save to file.

    Args:
        mesh: Node coordinates.
        solution: Field values at nodes.
        output_path: Path to save the figure.
        title: Plot title.
        xlabel, ylabel: Axis labels.

    Returns:
        Path to the saved figure.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
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
    """Solve a unified 1D problem and save a plot of the solution."""
    sol = solve(problem, method=method, n=n)
    path = plot_solution(
        mesh=sol["mesh"],
        solution=sol["solution"],
        output_path=output_path,
        title=f"{problem.name} ({method}, n={n})",
        xlabel=str(problem.domain.coordinates[0]) if problem.domain else "x",
        ylabel=next(iter(problem.fields.keys()), "u"),
    )
    return {
        "plot_path": str(path),
        **sol,
    }
