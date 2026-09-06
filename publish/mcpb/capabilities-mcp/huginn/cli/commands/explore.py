"""Exploration mode command."""

from __future__ import annotations

import asyncio

import click
from rich.panel import Panel

from huginn.cli.context import CliContext


@click.command()
@click.argument("objective")
@click.option(
    "--strategy",
    "-s",
    default="pareto",
    type=click.Choice(["pareto", "bayesian", "grid"]),
)
@click.option("--max-branches", "-b", default=10, help="Maximum parallel branches")
@click.option(
    "--max-iterations", "-i", default=20, help="Maximum exploration iterations"
)
@click.pass_obj
def explore(
    ctx: CliContext,
    objective: str,
    strategy: str,
    max_branches: int,
    max_iterations: int,
) -> None:
    """Enter exploration mode to systematically search a design space."""
    from huginn.exploration.orchestrator import ExplorationOrchestrator
    from huginn.exploration.strategies import ParetoPruningStrategy

    ctx.console.print(
        Panel(
            f"[bold green]Exploration Mode[/bold green]\n"
            f"Objective: {objective}\n"
            f"Strategy: {strategy}\n"
            f"Max branches: {max_branches}\n"
            f"Max iterations: {max_iterations}",
            title="Exploration",
            border_style="green",
        )
    )

    cfg = ctx.load_config()
    try:
        orch = ExplorationOrchestrator(
            strategy=ParetoPruningStrategy(max_active=max_branches),
            max_parallel=min(cfg.max_parallel_branches, max_branches),
        )
        result = asyncio.run(
            orch.explore(
                objective=objective,
                initial_branches=[
                    {"name": "baseline", "hypothesis": f"Baseline for: {objective}"}
                ],
                objectives_config={"score": "maximize"},
                max_iterations=max_iterations,
            )
        )
        ctx.console.print(
            f"[green]✓[/green] Exploration complete: {result.convergence_reason}"
        )
        ctx.console.print(f"  Branches explored: {result.n_branches_explored}")
        ctx.console.print(f"  Branches pruned: {result.n_branches_pruned}")
        ctx.console.print(f"  Pareto front size: {len(result.pareto_front)}")
        if result.best_branch:
            ctx.console.print(f"  Best branch: {result.best_branch['name']}")
    except Exception as e:
        ctx.console.print(f"[red]Exploration failed: {e}[/red]")
