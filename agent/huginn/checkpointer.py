"""Persistent checkpointing for HuginnAgent conversations.

By default LangGraph agents use an in-memory checkpointer, which means all
conversation state is lost when the process restarts. This module provides a
small factory that switches to SQLite persistence when a path is configured.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator


@contextmanager
def persistent_checkpointer(
    path: str | Path | None = None,
) -> Generator[Any, None, None]:
    """Context manager yielding a SQLite-backed SqliteSaver.

    The database connection is closed when the context exits, preventing
    resource leaks in long-running processes.
    """
    from langgraph.checkpoint.sqlite import SqliteSaver

    if path is None:
        path = os.environ.get("HUGINN_CHECKPOINTER_PATH")
    if path is None:
        default = Path.home() / ".huginn" / "checkpoints.sqlite"
        default.parent.mkdir(parents=True, exist_ok=True)
        path = default

    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    with SqliteSaver.from_conn_string(str(path)) as saver:
        yield saver


def create_checkpointer(path: str | Path | None = None) -> Any:
    """Create a LangGraph checkpointer.

    * ``path`` is given -> SQLite-backed persistent checkpointer.
    * ``path`` is ``":memory:"`` -> SQLite in-memory checkpointer.
    * ``path`` is None -> use ``HUGINN_CHECKPOINTER_PATH`` env var if set,
      otherwise a default SQLite file under ``~/.huginn/checkpoints.sqlite``.

    The returned object is a ``langgraph.checkpoint.sqlite.SqliteSaver``.
    """
    if path is None:
        path = os.environ.get("HUGINN_CHECKPOINTER_PATH")
    if path is None:
        default = Path.home() / ".huginn" / "checkpoints.sqlite"
        default.parent.mkdir(parents=True, exist_ok=True)
        path = default

    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    from langgraph.checkpoint.sqlite import SqliteSaver

    # ``from_conn_string`` is a context-manager factory; we enter it once and
    # keep the saver alive for the lifetime of the agent.
    cm = SqliteSaver.from_conn_string(str(path))
    saver = cm.__enter__()
    return saver


def create_in_memory_checkpointer() -> Any:
    """Create an in-memory checkpointer for tests or ephemeral agents."""
    from langgraph.checkpoint.memory import InMemorySaver

    return InMemorySaver()
