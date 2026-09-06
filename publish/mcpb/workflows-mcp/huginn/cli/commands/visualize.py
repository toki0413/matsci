"""Visualization commands for evolution/benchmark/exploration results."""

from __future__ import annotations

from pathlib import Path

import click

from huginn.cli.context import CliContext
from huginn.visualize import plot_from_file


@click.group(name="visualize")
@click.pass_obj
def visualize(ctx: CliContext) -> None:
    """Visualize benchmark, evolution, or exploration results."""


@visualize.command("bench")
@click.argument("report")
@click.option("--output", "-o", help="Output image path (default: <report>.png)")
@click.option(
    "--type",
    "plot_type",
    type=click.Choice(["bar", "pie"]),
    default="bar",
    help="Plot style",
)
@click.pass_obj
def visualize_bench(
    ctx: CliContext,
    report: str,
    output: str | None,
    plot_type: str,
) -> None:
    """Visualize a benchmark report JSON."""
    output_path = _resolve_output(report, output, "png")
    plot_from_file("bench", report, output_path, plot_type=plot_type)
    ctx.console.print(f"[green]✓[/green] Benchmark plot saved to {output_path}")


@visualize.command("evolution")
@click.argument("report")
@click.option("--output", "-o", help="Output image path (default: <report>.png)")
@click.option(
    "--type",
    "plot_type",
    type=click.Choice(["summary", "confidence", "convergence"]),
    default="summary",
    help=(
        "Plot style. 'summary' = counts + confidence; "
        "'confidence' = confidence histogram only; "
        "'convergence' expects an evolution_history.json list"
    ),
)
@click.pass_obj
def visualize_evolution(
    ctx: CliContext,
    report: str,
    output: str | None,
    plot_type: str,
) -> None:
    """Visualize an evolution report JSON."""
    output_path = _resolve_output(report, output, "png")
    plot_from_file("evolution", report, output_path, plot_type=plot_type)
    ctx.console.print(f"[green]✓[/green] Evolution plot saved to {output_path}")


@visualize.command("explore")
@click.argument("result")
@click.option("--output", "-o", help="Output image path (default: <result>.png)")
@click.option(
    "--type",
    "plot_type",
    type=click.Choice(["auto", "2d", "3d", "parallel", "radar"]),
    default="auto",
    help="Plot style",
)
@click.pass_obj
def visualize_explore(
    ctx: CliContext,
    result: str,
    output: str | None,
    plot_type: str,
) -> None:
    """Visualize an exploration result JSON."""
    output_path = _resolve_output(result, output, "png")
    plot_from_file("explore", result, output_path, plot_type=plot_type)
    ctx.console.print(f"[green]✓[/green] Exploration plot saved to {output_path}")


def _resolve_output(input_path: str, output: str | None, suffix: str) -> Path:
    if output:
        return Path(output)
    base = Path(input_path)
    # Replace original extension or append suffix
    return base.with_suffix(f".{suffix}")
