"""Plotting command for tabular data (CSV/JSON).

A lightweight multimodal visualization adapter: turn agent-generated data
files into matplotlib figures without leaving the CLI.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import click

from huginn.cli.context import CliContext


@click.command("plot")
@click.argument("data_file", type=click.Path(exists=True, path_type=Path))
@click.option("--x", "x_column", help="Column/field to use as x-axis")
@click.option("--y", "y_column", required=True, help="Column/field to use as y-axis")
@click.option(
    "--kind",
    type=click.Choice(["line", "scatter", "bar", "hist"]),
    default="line",
    help="Plot kind",
)
@click.option(
    "--output", "-o", type=click.Path(path_type=Path), help="Output image path"
)
@click.option("--title", default="", help="Plot title")
@click.option("--xlabel", default="", help="X-axis label")
@click.option("--ylabel", default="", help="Y-axis label")
@click.option("--dpi", default=150, type=int, help="Image resolution")
@click.pass_obj
def plot(
    ctx: CliContext,
    data_file: Path,
    x_column: str | None,
    y_column: str,
    kind: str,
    output: Path | None,
    title: str,
    xlabel: str,
    ylabel: str,
    dpi: int,
) -> None:
    """Generate a plot from a CSV or JSON data file."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        ctx.console.print(
            "[red]Plotting requires matplotlib. Install: pip install matplotlib[/red]"
        )
        raise click.ClickException(str(e)) from e

    suffix = data_file.suffix.lower()
    if suffix == ".csv":
        records = _load_csv(data_file)
    elif suffix in (".json", ".jsonl"):
        records = _load_json(data_file)
    else:
        ctx.console.print(
            f"[red]Unsupported file format: {suffix}. Use .csv or .json.[/red]"
        )
        raise click.Abort()

    if not records:
        ctx.console.print("[yellow]No data records found.[/yellow]")
        return

    x_values, y_values = _extract_series(records, x_column, y_column)
    if y_values is None:
        ctx.console.print(
            f"[red]Y column '{y_column}' not found or contains no numeric data.[/red]"
        )
        raise click.Abort()

    fig, ax = plt.subplots(figsize=(8, 5))
    if kind == "line":
        ax.plot(x_values or range(len(y_values)), y_values, marker="o")
    elif kind == "scatter":
        ax.scatter(x_values or range(len(y_values)), y_values)
    elif kind == "bar":
        ax.bar(
            [str(v) for v in (x_values or range(len(y_values)))],
            y_values,
        )
        ax.tick_params(axis="x", rotation=45)
    elif kind == "hist":
        ax.hist(y_values, bins=min(20, max(5, len(y_values) // 5)))

    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    elif y_column:
        ax.set_ylabel(y_column)

    fig.tight_layout()

    if output is None:
        output = ctx.workspace / f"{data_file.stem}_{kind}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    ctx.console.print(f"[green]✓[/green] Plot saved to {output}")


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _load_json(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    return []


def _extract_series(
    records: list[dict[str, Any]],
    x_column: str | None,
    y_column: str,
) -> tuple[list[Any] | None, list[float] | None]:
    y_values: list[float] = []
    x_values: list[Any] | None = [] if x_column else None
    for row in records:
        y_raw = row.get(y_column)
        if y_raw is None:
            continue
        try:
            y_values.append(float(y_raw))
        except (TypeError, ValueError):
            continue
        if x_values is not None:
            x_values.append(row.get(x_column))
    if not y_values:
        return x_values, None
    return x_values, y_values
