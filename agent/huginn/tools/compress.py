"""Tool output compression for the LLM context window.

Large tool outputs (DFT energies, MD trajectories, FEM meshes, logs) can
quickly blow up the prompt. This module provides opinionated, lossy
compression that preserves the information an LLM actually needs:
success/failure status, key scalars, short summaries, and error tails.
"""

from __future__ import annotations

from typing import Any

from huginn.utils.tokens import rough_token_count_for_text


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
