"""Benchmark runner command."""

from __future__ import annotations

import click
from rich.panel import Panel

from huginn.cli.context import CliContext
from huginn.config import HuginnConfig


@click.command()
@click.option("--evolve", is_flag=True, help="Run evolution cycle after benchmarking")
@click.option("--categories", "-c", help="Comma-separated categories to run")
@click.option("--output", "-o", default="bench_report.json", help="Report output path")
@click.pass_obj
def bench(ctx: CliContext, evolve: bool, categories: str | None, output: str) -> None:
    """Run the benchmark suite and optionally trigger self-evolution."""
    from huginn.bench.runner import BenchmarkRunner

    cfg = (
        HuginnConfig.load(ctx.config_path)
        if ctx.config_path
        else HuginnConfig.from_env()
    )
    runner = BenchmarkRunner(config=cfg)
    cats = (
        [c.strip() for c in categories.split(",") if c.strip()] if categories else None
    )

    report = runner.run(evolve=evolve, categories=cats)
    report_path = ctx.workspace / output
    runner.save_report(report, report_path)

    ctx.console.print(
        Panel(
            f"[bold blue]Benchmark Report[/bold blue]\n"
            f"Run ID: {report.run_id}\n"
            f"Total: {report.total}  Passed: {report.passed}  Failed: {report.failed}  Skipped: {report.skipped}\n"
            f"Pass rate: {report.metrics.get('pass_rate', 0):.0%}\n"
            f"Avg task time: {report.metrics.get('avg_task_time_seconds', 0):.2f}s\n"
            f"Report saved to: {report_path}",
            title="Bench",
            border_style="blue",
        )
    )

    for r in report.results:
        icon = "[green]✓[/green]" if r.passed else "[red]✗[/red]"
        ctx.console.print(f"{icon} {r.task_id}: {r.reason}")
