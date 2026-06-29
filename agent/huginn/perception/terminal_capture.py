"""Terminal capture — capture and parse huginn's own terminal output.

Extracts ANSI sequences, tracks command status, detects progress bars,
error patterns, and convergence states from tool execution output.

Usage:
    capture = TerminalCapture()
    capture.feed(b"\x1b[32mSuccess\x1b[0m\n")
    status = capture.get_status()
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TerminalStatus:
    """Parsed state from terminal output stream."""
    current_command: str | None = None
    last_output: str = ""
    error_detected: bool = False
    warning_count: int = 0
    progress_percent: float | None = None
    is_running: bool = False
    ansi_styles: dict[str, str] = field(default_factory=dict)


class TerminalCapture:
    """Capture and parse ANSI terminal output for state extraction.

    No external dependencies. Pure Python regex-based ANSI parsing.
    """

    # ANSI escape sequences
    ANSI_RE = re.compile(
        r"\x1b\[(\d+(?:;\d+)*)m|\x1b\[([\d;]*)[A-Za-z]"
    )
    PROGRESS_RE = re.compile(
        r"(\d{1,3})%|(\d{1,3})\s*/\s*(\d+)\s*\[.*?\]"
    )
    ERROR_RE = re.compile(
        r"(?i)(error|exception|traceback|failed|fatal|\bERR\b|\bFAIL\b)"
    )
    WARNING_RE = re.compile(
        r"(?i)(warning|warn|deprecat|\bWARN\b)"
    )

    # Known convergence patterns (material science simulation output)
    CONVERGED_RE = re.compile(
        r"(?i)(converged|convergence\s*reached|scf\s*converged|optimization\s*complete)"
    )
    NOT_CONVERGED_RE = re.compile(
        r"(?i)(not\s*converged|convergence\s*failed|scf\s*not\s*converged)"
    )

    def __init__(self) -> None:
        self._buffer = ""
        self._status = TerminalStatus()
        self._lines: list[str] = []

    def feed(self, data: bytes | str) -> None:
        """Feed raw terminal bytes into the parser."""
        text = data.decode("utf-8", errors="replace") if isinstance(data, bytes) else data
        self._buffer += text

        # Extract complete lines
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._process_line(line)

    def _process_line(self, line: str) -> None:
        """Parse a single line of terminal output."""
        # Strip ANSI for analysis but keep original for display
        clean = self._strip_ansi(line)
        self._lines.append(clean)
        if len(self._lines) > 1000:
            self._lines = self._lines[-500:]  # Keep last 500 lines

        self._status.last_output = clean

        # Detect errors
        if self.ERROR_RE.search(clean):
            self._status.error_detected = True

        # Count warnings
        if self.WARNING_RE.search(clean):
            self._status.warning_count += 1

        # Detect progress
        match = self.PROGRESS_RE.search(clean)
        if match:
            if match.group(1):
                self._status.progress_percent = float(match.group(1))
            elif match.group(2) and match.group(3):
                self._status.progress_percent = (
                    float(match.group(2)) / float(match.group(3)) * 100
                )

        # Detect convergence (material science specific)
        if self.CONVERGED_RE.search(clean):
            self._status.is_running = False
        elif self.NOT_CONVERGED_RE.search(clean):
            self._status.error_detected = True
            self._status.is_running = False

        # Detect command prompt (shell-like)
        if clean.strip().startswith(">>>") or clean.strip().endswith("$"):
            self._status.is_running = False

        # Detect running state from progress indicators
        if self._status.progress_percent is not None and self._status.progress_percent < 100:
            self._status.is_running = True

    def _strip_ansi(self, text: str) -> str:
        """Remove ANSI escape sequences from text."""
        return self.ANSI_RE.sub("", text)

    def get_status(self) -> TerminalStatus:
        """Return current parsed status."""
        return self._status

    def get_recent_lines(self, n: int = 50) -> list[str]:
        """Return last N parsed lines."""
        return self._lines[-n:]

    def reset(self) -> None:
        """Clear all captured state."""
        self._buffer = ""
        self._lines.clear()
        self._status = TerminalStatus()
