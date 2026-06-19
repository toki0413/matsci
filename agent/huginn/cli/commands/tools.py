"""List available tools."""

from __future__ import annotations

import click
from rich.panel import Panel

from huginn.cli.context import console
from huginn.tools import register_all_tools
from huginn.tools.registry import ToolRegistry


@click.command()
def tools() -> None:
    """List all available tools."""
    register_all_tools()

    console.print(
        Panel(
            f"[bold blue]Available Tools ({len(ToolRegistry.list_tools())})[/bold blue]",
            border_style="blue",
        )
    )

    for name in ToolRegistry.list_tools():
        tool = ToolRegistry.get(name)
        if tool:
            try:
                is_ro = hasattr(tool, "is_read_only") and tool.is_read_only(None)
            except Exception:
                is_ro = False
            read_only = "[green]read-only[/green]" if is_ro else ""
            console.print(
                f"  [bold]{name}[/bold] — {tool.description[:60]}... {read_only}"
            )
