"""Telemetry summary command."""

from __future__ import annotations

import click
from rich.panel import Panel

from huginn.agent import HuginnAgent
from huginn.cli.context import CliContext


@click.command()
@click.pass_obj
def telemetry(ctx: CliContext) -> None:
    """Show telemetry summary for the default agent profile."""
    cfg = ctx.load_config()
    agent = HuginnAgent.from_config(cfg)
    try:
        summary = agent.telemetry_summary()
        ctx.console.print(
            Panel(
                f"[bold blue]Telemetry Summary[/bold blue]\n"
                f"Total spans: {summary.get('total_spans', 0)}",
                border_style="blue",
            )
        )
        for name, info in summary.get("by_name", {}).items():
            ctx.console.print(
                f"  {name}: count={info['count']} duration_ms={info['duration_ms']:.1f}"
            )
    finally:
        agent.close()
