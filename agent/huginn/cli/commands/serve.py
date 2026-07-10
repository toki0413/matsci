"""Start the HTTP/WebSocket server."""

from __future__ import annotations

import atexit
import contextlib
import logging
import os
import signal
import sys
import threading
from pathlib import Path
from typing import Any

import click
from rich.panel import Panel

from huginn.cli.context import CliContext
from huginn.tools import register_all_tools
from huginn.tools.registry import ToolRegistry
logger = logging.getLogger(__name__)


_log = logging.getLogger("huginn.serve")

# Filled in by serve() once the uvicorn Server is built, so the signal
# handler can flip should_exit instead of yanking the process out from
# under in-flight requests.
_uvicorn_server: Any = None
# Latches on the first drain signal — a second one forces an immediate exit.
_shutting_down: bool = False


def _write_pid_file(port: int) -> Path:
    """Write PID file so the parent (Rust) can track us."""
    pid_dir = Path(os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))) / "Huginn"
    pid_dir.mkdir(parents=True, exist_ok=True)
    pid_file = pid_dir / f"server-{port}.pid"
    pid_file.write_text(str(os.getpid()))
    return pid_file


def _setup_signal_handlers(pid_file: Path) -> None:
    """Register SIGINT/SIGTERM to drain in-flight requests before exit.

    The old handler called sys.exit(0) directly, which stranded whatever
    requests were mid-flight and left the SQLite connections (research_log,
    anomaly_log, campaign store) dangling. Now we ask uvicorn to stop
    accepting new connections and let its lifespan shutdown close the
    databases and cancel tasks. A second signal forces a hard exit so
    operators always have an escape hatch.
    """

    def _drain(signum, frame):
        global _shutting_down
        # Second hit while already draining — bail out hard. Anyone running
        # this in production has a way to force-kill if the drain hangs.
        if _shutting_down:
            _log.warning("second signal received, forcing immediate exit")
            try:
                pid_file.unlink(missing_ok=True)
            except Exception:
                logger.debug("unlink failed", exc_info=True)
            sys.exit(0)

        _shutting_down = True
        _log.info("Shutting down, waiting for in-flight requests...")

        server = _uvicorn_server
        if server is not None:
            # Stop accepting new connections; uvicorn finishes what's in
            # flight, then runs the lifespan shutdown (where SQLite stores
            # get closed and agent tasks cancelled).
            server.should_exit = True
            return

        # No server bound yet — the signal caught us during startup. Give
        # the process a short grace window to either finish booting or
        # abort, then exit so we don't hang on the import lock.
        def _force_exit_after_timeout():
            import time

            time.sleep(5.0)
            try:
                pid_file.unlink(missing_ok=True)
            except Exception:
                logger.debug("unlink failed", exc_info=True)
            sys.exit(0)

        threading.Thread(target=_force_exit_after_timeout, daemon=True).start()

    signal.signal(signal.SIGINT, _drain)
    signal.signal(signal.SIGTERM, _drain)
    # Windows sends SIGBREAK on Ctrl+Break; treat it the same as SIGTERM.
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _drain)  # type: ignore[attr-defined]
    atexit.register(lambda: pid_file.unlink(missing_ok=True))


def _build_drainable_server(app: Any, host: str, port: int):
    """Construct a uvicorn Server whose signal handling we control.

    uvicorn's default capture_signals() swaps its own handle_exit in for
    SIGINT/SIGTERM, which would mask the drain handler installed above and
    re-raise it after shutdown (causing a confusing second fire). Subclassing
    to no-op the swap keeps _drain as the single source of truth.
    """
    import uvicorn

    class _DrainableServer(uvicorn.Server):
        @contextlib.contextmanager
        def capture_signals(self):  # type: ignore[override]
            yield

    # Long LLM calls (DeepSeek reasoner 60s+) outlast uvicorn's default
    # 20s WS ping timeout. 5 min keeps the connection alive mid-response.
    config = uvicorn.Config(app, host=host, port=port,
                            ws_ping_interval=300, ws_ping_timeout=300)
    return _DrainableServer(config)


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
        import uvicorn  # noqa: F401  (import check preserved for the error path)

        from huginn.server import app

        global _uvicorn_server
        _uvicorn_server = _build_drainable_server(app, host=host, port=port)
        _uvicorn_server.run()
    except ImportError:
        ctx.console.print(
            "[red]uvicorn not installed. Run: pip install uvicorn fastapi[/red]"
        )
    finally:
        pid_file.unlink(missing_ok=True)
