"""Start the HTTP/WebSocket server."""

from __future__ import annotations

import click
from rich.panel import Panel

from huginn.cli.context import CliContext
from huginn.tools import register_all_tools
from huginn.tools.registry import ToolRegistry


@click.command()
@click.option("--port", "-p", default=8000, help="Server port")
@click.option("--host", "-h", default="127.0.0.1", help="Server host")
@click.pass_obj
def serve(ctx: CliContext, port: int, host: str) -> None:
    """Start the HTTP/WebSocket server for the desktop app."""
    register_all_tools()

    ctx.console.print(
        Panel(
            f"[bold blue]Huginn Server[/bold blue]\n"
            f"URL: http://{host}:{port}\n"
            f"WebSocket: ws://{host}:{port}/ws/agent\n"
            f"Tools: {len(ToolRegistry.list_tools())}",
            title="Server",
            border_style="blue",
        )
    )

    try:
        import uvicorn

        from huginn.server import app

        uvicorn.run(app, host=host, port=port)
    except ImportError:
        ctx.console.print(
            "[red]uvicorn not installed. Run: pip install uvicorn fastapi[/red]"
        )
