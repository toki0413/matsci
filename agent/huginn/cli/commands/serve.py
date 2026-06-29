"""Start the HTTP/WebSocket server."""

from __future__ import annotations

import atexit
import os
import signal
import sys
from pathlib import Path

import click
from rich.panel import Panel

from huginn.cli.context import CliContext
from huginn.tools import register_all_tools
from huginn.tools.registry import ToolRegistry


def _write_pid_file(port: int) -> Path:
    """Write PID file so the parent (Rust) can track us."""
    pid_dir = Path(os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))) / "Huginn"
    pid_dir.mkdir(parents=True, exist_ok=True)
    pid_file = pid_dir / f"server-{port}.pid"
    pid_file.write_text(str(os.getpid()))
    return pid_file


def _setup_signal_handlers(pid_file: Path) -> None:
    """Register SIGINT/SIGTERM to remove PID file and exit cleanly."""

    def _cleanup(signum, frame):
        try:
            pid_file.unlink(missing_ok=True)
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)
    atexit.register(lambda: pid_file.unlink(missing_ok=True))


@click.command()
@click.option("--port", "-p", default=8001, help="Server port")
@click.option("--host", "-h", default="127.0.0.1", help="Server host")
@click.pass_obj
def serve(ctx: CliContext, port: int, host: str) -> None:
    """Start the HTTP/WebSocket server for the desktop app."""
    register_all_tools()

    pid_file = _write_pid_file(port)
    _setup_signal_handlers(pid_file)

    ctx.console.print(
        Panel(
            f"[bold blue]Huginn Server[/bold blue]\n"
            f"URL: http://{host}:{port}\n"
            f"WebSocket: ws://{host}:{port}/ws/agent\n"
            f"PID: {os.getpid()}\n"
            f"PID file: {pid_file}\n"
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
    finally:
        pid_file.unlink(missing_ok=True)
