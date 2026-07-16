"""Benchmark runner command.

6 个能力层对标社区 AI Scientist 评测:
  general / physics / lineage / repro / optim / research
research 多 trial 走独立脚本, 其余 5 个走本命令.
"""

from __future__ import annotations

import click

from huginn.cli.context import CliContext
from huginn.cli.design_system import get_design_system
from huginn.config import HuginnConfig

SUITES = [
    "general", "mmlu", "sciq", "arc",
    "gpqa", "cmmlu", "mmlu_pro", "external",
    "physics", "lineage", "repro", "optim", "research",
]


@click.command()
@click.option("--suite", "-s", default="general", type=click.Choice(SUITES),
              help="Benchmark suite: general/mmlu/sciq/arc/gpqa/cmmlu/mmlu_pro/external/physics/lineage/repro/optim/research")
@click.option("--evolve", is_flag=True, help="Run evolution cycle after benchmarking (general only)")
@click.option("--categories", "-c", help="Comma-separated categories to run (general only)")
@click.option("--max-tasks", "-n", type=int, default=None, help="Limit task count (for mmlu/sciq/arc)")
@click.option("--output", "-o", default="bench_report.json", help="Report output path")
@click.pass_obj
def bench(ctx: CliContext, suite: str, evolve: bool, categories: str | None,
          max_tasks: int | None, output: str) -> None:
    """Run benchmark suite against Huginn agent (DeepSeek v4-flash)."""
    ds = get_design_system()
    cfg = (
        HuginnConfig.load(ctx.config_path)
        if ctx.config_path
        else HuginnConfig.from_env()
    )

    if suite == "research":
        # ResearchClawBench 多 trial (11 题 × 3), 走独立脚本
        ds.dialog(
            title="Bench",
            content=(
                "[yellow]ResearchClawBench 需要多 trial, 请用独立脚本:[/yellow]\n"
                "  python tests/test_clawbench_runner.py --trials 3"
            ),
        )
        return

    from huginn.bench import BenchmarkRunner, get_suite_tasks

    tasks = get_suite_tasks(suite, max_tasks=max_tasks)
    if not tasks:
        ds.error(f"未知 suite: {suite}")
        return

    # ponytail: bench 命令注入 MemoryManager; 升级路径是 bench 复用 live agent 的 memory
    try:
        from huginn.memory.manager import MemoryManager
        mem = MemoryManager()
    except Exception:
        mem = None

    runner = BenchmarkRunner(tasks=tasks, config=cfg, memory_manager=mem)
    cats = (
        [c.strip() for c in categories.split(",") if c.strip()] if categories else None
    )

    ds.dialog(title="Bench", content=f"[bold blue]Running suite: {suite} ({len(tasks)} tasks)[/bold blue]")

    report = runner.run(evolve=evolve and suite == "general", categories=cats)
    report_path = ctx.workspace / output
    runner.save_report(report, report_path)

    ds.dialog(
        title="Bench",
        content=(
            f"[bold blue]Benchmark Report ({suite})[/bold blue]\n"
            f"Run ID: {report.run_id}\n"
            f"Total: {report.total}  Passed: {report.passed}  Failed: {report.failed}  Skipped: {report.skipped}\n"
            f"Pass rate: {report.metrics.get('pass_rate', 0):.0%}\n"
            f"Avg task time: {report.metrics.get('avg_task_time_seconds', 0):.2f}s\n"
            f"Report saved to: {report_path}"
        ),
    )

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
