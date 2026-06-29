"""Tool output compression for the LLM context window.

Large tool outputs (DFT energies, MD trajectories, FEM meshes, logs) can
quickly blow up the prompt. This module provides opinionated, lossy
compression that preserves the information an LLM actually needs:
success/failure status, key scalars, short summaries, and error tails.

The ``smart_compress_text`` function adds LLM-based summarization of the
middle portion of long outputs, preserving research-relevant context that
pure head/tail truncation would discard.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from huginn.utils.tokens import count_tokens, rough_token_count_for_text

logger = logging.getLogger(__name__)


class ToolOutputCompressor:
    """Compress tool outputs to fit a target token budget."""

    def __init__(
        self,
        max_output_tokens: int = 8000,
        max_text_lines: int = 40,
        array_head_tail: int = 4,
        keep_keys: set[str] | None = None,
    ):
        self.max_output_tokens = max_output_tokens
        self.max_text_lines = max_text_lines
        self.array_head_tail = array_head_tail
        self.keep_keys = keep_keys or {
            "success",
            "error",
            "energy",
            "total_energy",
            "band_gap",
            "lattice_constant",
            "converged",
            "iteration",
            "message",
            "summary",
        }

    def compress(self, data: Any) -> Any:
        """Recursively compress ``data`` in-place style (returns a new tree)."""
        if isinstance(data, str):
            return self._compress_text(data)
        if isinstance(data, (int, float, bool)) or data is None:
            return data
        if isinstance(data, list):
            return self._compress_list(data)
        if isinstance(data, dict):
            return self._compress_dict(data)
        # Fallback: stringify and truncate.
        return self._compress_text(str(data))

    def _compress_text(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)

        lines = text.splitlines()
        if len(lines) > self.max_text_lines:
            head = lines[: self.max_text_lines // 2]
            tail = (
                lines[-self.max_text_lines // 2 :] if self.max_text_lines // 2 else []
            )
            text = "\n".join(
                head
                + [f"... ({len(lines) - len(head) - len(tail)} lines omitted) ..."]
                + tail
            )

        # If the string is now within token budget, leave it alone.
        if rough_token_count_for_text(text) <= self.max_output_tokens:
            return text

        # Final char budget (approx 4 chars/token).
        max_chars = self.max_output_tokens * 4
        if len(text) > max_chars:
            text = text[:max_chars] + "\n...[truncated]"
        return text

    def _compress_list(self, items: list[Any]) -> list[Any] | dict[str, Any]:
        # Numeric arrays: summarize + head/tail.
        if items and all(isinstance(x, (int, float)) for x in items):
            return self._summarize_numeric(items)

        compressed = [self.compress(v) for v in items]
        if len(compressed) > self.array_head_tail * 2 + 1:
            head = compressed[: self.array_head_tail]
            tail = compressed[-self.array_head_tail :]
            return {
                "_type": "compressed_list",
                "count": len(items),
                "head": head,
                "tail": tail,
            }
        return compressed

    def _summarize_numeric(self, values: list[float]) -> dict[str, Any]:
        clean = [float(v) for v in values if v is not None]
        if not clean:
            return {"count": len(values), "values": []}
        return {
            "count": len(values),
            "min": round(min(clean), 6),
            "max": round(max(clean), 6),
            "mean": round(sum(clean) / len(clean), 6),
            "head": [round(v, 6) for v in clean[: self.array_head_tail]],
            "tail": [round(v, 6) for v in clean[-self.array_head_tail :]],
        }

    def _compress_dict(self, data: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in data.items():
            # Always preserve small critical keys.
            if key in self.keep_keys:
                result[key] = (
                    value if not isinstance(value, str) else self._compress_text(value)
                )
                continue

            # Large string values under unknown keys get summarized.
            if (
                isinstance(value, str)
                and rough_token_count_for_text(value) > self.max_output_tokens
            ):
                result[key] = self._compress_text(value)
                continue

            result[key] = self.compress(value)
        return result


def compress_tool_output(
    data: Any,
    max_output_tokens: int | None = None,
    keep_keys: set[str] | None = None,
) -> Any:
    """Convenience entry point used by ToolAdapter."""
    compressor = ToolOutputCompressor(
        max_output_tokens=max_output_tokens or 8000,
        keep_keys=keep_keys,
    )
    return compressor.compress(data)


async def smart_compress_text(
    text: str,
    max_tokens: int = 8000,
    summarizer: Callable[[str], Any] | None = None,
) -> str:
    """Compress long text with optional LLM summarization of the middle.

    Strategy:
    1. If text fits in budget, return as-is.
    2. If a summarizer is available, keep first 20% + last 20% verbatim
       and replace the middle 60% with an LLM-generated summary.
    3. If no summarizer, fall back to head/tail truncation.

    Args:
        text: The text to compress.
        max_tokens: Target token budget.
        summarizer: Async callable that takes a text string and returns
            a summary. If None, falls back to head/tail truncation.
    """
    if not text:
        return text

    token_count = count_tokens(text)
    if token_count <= max_tokens:
        return text

    lines = text.splitlines(keepends=True)
    if len(lines) < 10:
        # Too few lines to meaningfully split — just truncate
        max_chars = max_tokens * 4
        return text[:max_chars] + "\n...[truncated]"

    # Split into head (20%), middle (60%), tail (20%)
    n = len(lines)
    head_end = max(1, n // 5)
    tail_start = n - max(1, n // 5)
    head = "".join(lines[:head_end])
    middle = "".join(lines[head_end:tail_start])
    tail = "".join(lines[tail_start:])

    if summarizer is not None and count_tokens(middle) > 200:
        try:
            result = await summarizer(middle)
            if hasattr(result, "content"):
                summary = result.content
            elif isinstance(result, dict):
                summary = result.get("content", str(result))
            else:
                summary = str(result)
            # Truncate summary to a reasonable size
            summary_tokens = count_tokens(summary)
            max_summary_tokens = max_tokens // 3
            if summary_tokens > max_summary_tokens:
                max_chars = max_summary_tokens * 4
                summary = summary[:max_chars] + "..."
            omitted = n - head_end - (n - tail_start)
            return (
                head
                + f"\n[... {omitted} lines summarized: {summary} ...]\n"
                + tail
            )
        except Exception as exc:
            logger.debug("Smart compression summarization failed: %s", exc)

    # Fallback: head/tail with omission marker
    omitted = n - head_end - (n - tail_start)
    return head + f"\n[... {omitted} lines omitted ...]\n" + tail
