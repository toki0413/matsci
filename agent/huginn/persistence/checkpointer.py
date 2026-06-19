"""Checkpointer backend abstraction and implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import suppress
from pathlib import Path
from typing import Any


class CheckpointerBackend(ABC):
    """Abstract backend for LangGraph conversation checkpoints."""

    @abstractmethod
    def get(self) -> Any:
        """Return a LangGraph-compatible checkpointer instance."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Release any resources held by the backend."""
        raise NotImplementedError


class SQLiteCheckpointerBackend(CheckpointerBackend):
    """SQLite-backed checkpointer using LangGraph's SqliteSaver."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = path
        self._saver: Any | None = None

    def get(self) -> Any:
        if self._saver is None:
            from langgraph.checkpoint.sqlite import SqliteSaver

            resolved = self._resolve_path()
            resolved.parent.mkdir(parents=True, exist_ok=True)
            self._saver = SqliteSaver.from_conn_string(str(resolved)).__enter__()
        return self._saver

    def close(self) -> None:
        if self._saver is not None:
            with suppress(Exception):
                self._saver.__exit__(None, None, None)
            self._saver = None

    def _resolve_path(self) -> Path:
        import os

        if self.path is not None:
            return Path(self.path).expanduser()
        env_path = os.environ.get("HUGINN_CHECKPOINTER_PATH")
        if env_path:
            return Path(env_path).expanduser()
        default = Path.home() / ".huginn" / "checkpoints.sqlite"
        default.parent.mkdir(parents=True, exist_ok=True)
        return default
