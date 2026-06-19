"""Export command — serialize Huginn records for sharing or archiving."""

from __future__ import annotations

import sys

import click
from rich.console import Console

from huginn.cli.context import CliContext
from huginn.export_manager import ExportManager


@click.command(name="export")
@click.option(
    "--source",
    "-s",
    required=True,
    type=click.Choice(["audit", "remote_jobs", "knowledge", "checkpoints"]),
    help="Data source to export",
)
@click.option(
    "--format",
    "-f",
    "fmt",
    default="json",
    type=click.Choice(["json", "markdown", "html"]),
    help="Export format",
)
@click.option(
    "--output",
    "-o",
    required=True,
    help="Output file path",
)
@click.pass_obj
def export_data(obj: CliContext, source: str, fmt: str, output: str) -> None:
    """Export Huginn records to JSON, Markdown, or HTML.

    Example:

        huginn export -s audit -f markdown -o audit_report.md
    """
    console = Console()
    manager = ExportManager(obj.workspace)

    try:
        result = manager.export(source=source, output_path=output, fmt=fmt)
    except Exception as exc:
        console.print(f"[red]Export failed:[/red] {exc}")
        sys.exit(1)

    console.print(
        f"[green]Exported {result.record_count} {source} record(s) to[/green] "
        f"{result.output_path}"
    )
