"""Benchmark runner command."""

from __future__ import annotations

import click

from huginn.cli.context import CliContext
from huginn.cli.design_system import get_design_system
from huginn.config import HuginnConfig


@click.command()
@click.option("--evolve", is_flag=True, help="Run evolution cycle after benchmarking")
@click.option("--categories", "-c", help="Comma-separated categories to run")
@click.option("--output", "-o", default="bench_report.json", help="Report output path")
@click.pass_obj
def bench(ctx: CliContext, evolve: bool, categories: str | None, output: str) -> None:
    """Run the benchmark suite and optionally trigger self-evolution."""
    from huginn.bench.runner import BenchmarkRunner

    ds = get_design_system()
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

    ds.dialog(
        title="Bench",
        content=(
            f"[bold blue]Benchmark Report[/bold blue]\n"
            f"Run ID: {report.run_id}\n"
            f"Total: {report.total}  Passed: {report.passed}  Failed: {report.failed}  Skipped: {report.skipped}\n"
            f"Pass rate: {report.metrics.get('pass_rate', 0):.0%}\n"
            f"Avg task time: {report.metrics.get('avg_task_time_seconds', 0):.2f}s\n"
            f"Report saved to: {report_path}"
        ),
    )

    # 进度可视化：按通过/失败比例渲染一条进度条
    ds.progress_bar(
        current=report.passed,
        total=report.total or 1,
        label="Passed",
    )

    for r in report.results:
        if r.passed:
            ds.success(f"{r.task_id}: {r.reason}")
        else:
            ds.error(f"{r.task_id}: {r.reason}")
