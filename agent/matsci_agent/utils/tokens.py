"""Rough token-count estimation utilities.

These are intentionally conservative fallbacks for when an API-based token
count is unavailable. Inspired by Claude Code's tokenEstimation.ts.
"""

from __future__ import annotations


def bytes_per_token_for_extension(ext: str | None) -> float:
    """Return a heuristic bytes-per-token ratio for dense formats."""
    if ext in ("json", "jsonl", "jsonc"):
        return 2.0
    return 4.0


def rough_token_count(text: str, bytes_per_token: float = 4.0) -> int:
    """Estimate token count from character length."""
    if not text:
        return 0
    return max(1, round(len(text) / bytes_per_token))


def rough_token_count_for_text(text: str, file_extension: str | None = None) -> int:
    """Estimate token count using a format-aware ratio."""
    return rough_token_count(text, bytes_per_token_for_extension(file_extension))
