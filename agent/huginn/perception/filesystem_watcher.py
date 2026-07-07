"""Filesystem watcher — real-time workspace monitoring for Huginn.

Replaces the naive file scanning in autoloop._perceive() with
inotify/watchdog-based event-driven detection.

Usage:
    watcher = FilesystemWatcher(workspace=".")
    watcher.start()
    for event in watcher.get_events(timeout=5.0):
        print(event)
    watcher.stop()
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import logging
logger = logging.getLogger(__name__)



@dataclass
class FileEvent:
    """A filesystem event."""
    event_type: str  # modified, created, deleted, moved
    path: str
    is_directory: bool
    timestamp: float
    src_path: str | None = None  # for moved events


class FilesystemWatcher:
    """Cross-platform filesystem watcher with graceful degradation.

    Uses watchdog when available (pip install watchdog).
    Falls back to periodic polling with mtime comparison.
    """

    def __init__(self, workspace: str | Path, patterns: list[str] | None = None):
        self.workspace = Path(workspace).resolve()
        self.patterns = patterns or ["*.py", "*.cif", "*.poscar", "*.vasp", "*.json", "*.md", "*.log"]
        self.events: list[FileEvent] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._watchdog_available = False
        self._observer: Any = None
        self._handlers: list[Callable[[FileEvent], None]] = []

    def start(self) -> None:
        """Start the watcher."""
        try:
            self._start_watchdog()
        except ImportError:
            self._start_polling()

    def stop(self) -> None:
        """Stop the watcher."""
        self._stop.set()
        if self._observer is not None:
            try:
                self._observer.stop()
                self._observer.join(timeout=2.0)
            except Exception:
                logger.debug("stop failed", exc_info=True)
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def on_event(self, handler: Callable[[FileEvent], None]) -> None:
        """Register an event handler."""
        self._handlers.append(handler)

    def get_events(self, timeout: float | None = None) -> list[FileEvent]:
        """Get accumulated events (non-blocking)."""
        with self._lock:
            events = self.events.copy()
            self.events.clear()
        return events

    def _emit(self, event: FileEvent) -> None:
        with self._lock:
            self.events.append(event)
        for handler in self._handlers:
            try:
                handler(event)
            except Exception:
                logger.debug("handler failed", exc_info=True)

    # ── Watchdog backend ───────────────────────────────────────

    def _start_watchdog(self) -> None:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler, FileSystemEvent

        class Handler(FileSystemEventHandler):
            def __init__(self, watcher: FilesystemWatcher) -> None:
                self.watcher = watcher

            def on_any_event(self, event: FileSystemEvent) -> None:
                if event.event_type in {"opened", "closed"}:
                    return
                e = FileEvent(
                    event_type=event.event_type,
                    path=str(event.src_path),
                    is_directory=event.is_directory,
                    timestamp=time.time(),
                    src_path=str(event.dest_path) if hasattr(event, "dest_path") else None,
                )
                self.watcher._emit(e)

        self._observer = Observer()
        self._observer.schedule(Handler(self), str(self.workspace), recursive=True)
        self._observer.start()
        self._watchdog_available = True

    # ── Polling backend ─────────────────────────────────────────

    def _start_polling(self) -> None:
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def _poll_loop(self) -> None:
        snapshot = self._snapshot()
        interval = 2.0  # seconds
        while not self._stop.wait(interval):
            new_snapshot = self._snapshot()
            self._diff(snapshot, new_snapshot)
            snapshot = new_snapshot

    def _snapshot(self) -> dict[str, float]:
        """Return {path: mtime} for tracked files."""
        result: dict[str, float] = {}
        for pattern in self.patterns:
            for path in self.workspace.rglob(pattern):
                try:
                    result[str(path)] = path.stat().st_mtime
                except OSError:
                    pass
        return result

    def _diff(self, old: dict[str, float], new: dict[str, float]) -> None:
        for path, mtime in new.items():
            if path not in old:
                self._emit(FileEvent("created", path, False, time.time()))
            elif old[path] != mtime:
                self._emit(FileEvent("modified", path, False, time.time()))
        for path in old:
            if path not in new:
                self._emit(FileEvent("deleted", path, False, time.time()))
