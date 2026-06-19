"""Self-evolution command."""

from __future__ import annotations

import click
from rich.panel import Panel

from huginn.cli.context import CliContext


@click.command()
@click.option("--logs-dir", help="Execution log directory to evolve from")
@click.pass_obj
def evolve(ctx: CliContext, logs_dir: str | None) -> None:
    """Run a self-evolution cycle from execution logs."""
    from huginn.evolution.engine import EvolutionEngine
    from huginn.evolution.logger import ExecutionLogger

    logger = ExecutionLogger(persist_dir=logs_dir) if logs_dir else ExecutionLogger()
    engine = EvolutionEngine(logger=logger)
    report = engine.run_full_evolution_cycle()

    ctx.console.print(
        Panel(
            f"[bold blue]Evolution Report[/bold blue]\n"
            f"Failure rules learned: {len(report['failure_rules'])}\n"
            f"Success skills extracted: {len(report['success_skills'])}\n"
            f"Prompt patches: {len(report['prompt_patches'])}\n"
            f"Total rules/skills: {report['total_rules_after']}/{report['total_skills_after']}",
            title="Evolve",
            border_style="blue",
        )
    )
