"""Long-term memory maintenance command."""

from __future__ import annotations

import click
from rich.panel import Panel

from huginn.agent import HuginnAgent
from huginn.checkpointer import create_in_memory_checkpointer
from huginn.cli.context import CliContext


@click.command("memory-maintenance")
@click.option(
    "--prune-threshold",
    type=float,
    default=None,
    help="Importance threshold for pruning",
)
@click.pass_obj
def memory_maintenance(ctx: CliContext, prune_threshold: float | None) -> None:
    """Run long-term memory decay, prune, and deduplication."""
    cfg = ctx.load_config()
    threshold = (
        prune_threshold
        if prune_threshold is not None
        else cfg.memory_decay_prune_threshold
    )
    agent = HuginnAgent(
        model=None, tools=[], checkpointer=create_in_memory_checkpointer()
    )
    try:
        summary = agent.memory.maintenance(prune_threshold=threshold)
        ctx.console.print(
            Panel(
                f"[bold blue]Memory Maintenance[/bold blue]\n"
                f"Decayed: {summary.get('decayed', 0)}\n"
                f"Pruned: {summary.get('pruned', 0)}\n"
                f"Expired: {summary.get('expired', 0)}\n"
                f"Deduplicated: {summary.get('deduplicated', 0)}",
                border_style="blue",
            )
        )
    finally:
        agent.close()
