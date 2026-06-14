"""Project-level context loader.

Reads a `.huginn.md` file from the workspace root and injects it into the
agent system prompt. Falls back to `AGENTS.md` if present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

DEFAULT_FILENAME = ".huginn.md"
FALLBACK_FILENAME = "AGENTS.md"


def _find_context_file(workspace: Path) -> Path | None:
    """Return the first existing project context file."""
    primary = workspace / DEFAULT_FILENAME
    if primary.exists() and primary.is_file():
        return primary
    fallback = workspace / FALLBACK_FILENAME
    if fallback.exists() and fallback.is_file():
        return fallback
    return None


def load_project_context(workspace: str | Path) -> str:
    """Load project context markdown from the workspace."""
    path = _find_context_file(Path(workspace))
    if path is None:
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def save_project_context(workspace: str | Path, content: str) -> dict:
    """Write project context markdown to `.huginn.md`."""
    path = Path(workspace) / DEFAULT_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return {"path": str(path), "bytes": len(content.encode("utf-8"))}


def project_context_path(workspace: str | Path) -> Path:
    """Return the primary project context file path."""
    return Path(workspace) / DEFAULT_FILENAME


def context_source(workspace: str | Path) -> Literal[".huginn.md", "AGENTS.md", "none"]:
    """Indicate which context file is being used."""
    primary = Path(workspace) / DEFAULT_FILENAME
    if primary.exists() and primary.is_file():
        return ".huginn.md"
    fallback = Path(workspace) / FALLBACK_FILENAME
    if fallback.exists() and fallback.is_file():
        return "AGENTS.md"
    return "none"
