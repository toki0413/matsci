"""List configured models and agent profiles."""

from __future__ import annotations

import click
from rich.panel import Panel

from huginn.cli.context import CliContext


@click.command("model-list")
@click.pass_obj
def model_list(ctx: CliContext) -> None:
    """List configured models and agent profiles."""
    cfg = ctx.load_config()
    ctx.console.print(
        Panel("[bold blue]Configured Models[/bold blue]", border_style="blue")
    )
    if cfg.models:
        for m in cfg.models:
            status = "[green]enabled[/green]" if m.enabled else "[red]disabled[/red]"
            ctx.console.print(
                f"  [bold]{m.alias}[/bold] {m.provider}:{m.model or 'auto'} ({status})"
            )
    else:
        ctx.console.print(
            f"  [dim]No model pool. Legacy provider: {cfg.provider} / {cfg.model or 'auto'}[/dim]"
        )

    ctx.console.print(
        Panel("[bold blue]Agent Profiles[/bold blue]", border_style="blue")
    )
    if cfg.agents:
        for a in cfg.agents:
            status = "[green]enabled[/green]" if a.enabled else "[red]disabled[/red]"
            ctx.console.print(
                f"  [bold]{a.id}[/bold] -> {a.model_alias} persona={a.persona} tools={len(a.tools)} ({status})"
            )
    else:
        ctx.console.print("  [dim]No agent profiles configured.[/dim]")
