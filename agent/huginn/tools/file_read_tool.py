"""File read tool — read text files with optional line range.

A read-only tool for inspecting source code, logs, and configuration files.
Safe to auto-execute.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult
from huginn.utils.tokens import rough_token_count_for_text

DEFAULT_MAX_SIZE_BYTES = 256 * 1024
DEFAULT_MAX_OUTPUT_TOKENS = 25000

# 二进制文件后缀 — read_text(errors="replace") 产出的乱码毫无用处还爆内存.
# agent 读 PDF 时乱码留在消息历史里, get_buffer_string 序列化时 MemoryError.
# ponytail: 后缀判断, 不读文件头. 天花板: 误判无后缀的二进制文件; 升级路径
# 读前 8 字节做 magic number 检测.
_BINARY_SUFFIXES = frozenset({
    ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp",
    ".zip", ".gz", ".tar", ".7z", ".rar", ".bz2", ".xz",
    ".exe", ".dll", ".so", ".dylib", ".bin", ".dat",
    ".mp4", ".avi", ".mov", ".mp3", ".wav", ".flac",
    ".pyc", ".pyo", ".class", ".jar",
    ".npy", ".npz", ".pkl", ".pt", ".pth", ".ckpt",
    ".db", ".sqlite", ".sqlite3",
})


class FileReadToolInput(BaseModel):
    action: Literal["read"] = Field(default="read")
    file_path: str = Field(..., description="Path to file")
    line_offset: int | None = Field(default=1, ge=1, description="1-based start line")
    n_lines: int | None = Field(default=None, description="Number of lines to read")
    tail_lines: int | None = Field(
        default=None, description="Read the last N lines instead of from line_offset"
    )
    max_size_bytes: int | None = Field(
        default=None, ge=1, description="Max file size in bytes"
    )
    max_output_tokens: int | None = Field(
        default=None, ge=1, description="Max output tokens"
    )
    working_dir: str | None = Field(default=None)


class FileReadTool(HuginnTool):
    """Read text files."""

    name = "file_read_tool"
    category = "core"
    description = "Read the contents of a text file, optionally with a line range."
    input_schema = FileReadToolInput

    def is_read_only(self, args: FileReadToolInput) -> bool:
        return True

    async def call(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        input_data = FileReadToolInput(**args)
        work_dir = (
            Path(input_data.working_dir) if input_data.working_dir else Path.cwd()
        )
        path = work_dir / input_data.file_path
        if not path.is_absolute():
            path = path.resolve()

        # Path traversal protection: restrict reads to the workspace directory.
        # Allow explicit opt-out via HUGINN_ALLOW_UNRESTRICTED_READ for power users.
        allow_unrestricted = os.environ.get(
            "HUGINN_ALLOW_UNRESTRICTED_READ", ""
        ).lower() in ("1", "true", "yes")
        if not allow_unrestricted:
            try:
                work_dir_resolved = work_dir.resolve()
                path.relative_to(work_dir_resolved)
            except ValueError:
                return ToolResult(
                    data=None,
                    success=False,
                    error=(
                        f"Access denied: {path} is outside the workspace "
                        f"({work_dir_resolved}). Set HUGINN_ALLOW_UNRESTRICTED_READ=1 "
                        "to override."
                    ),
                )

        if not path.exists():
            return ToolResult(data=None, success=False, error=f"File not found: {path}")
        if not path.is_file():
            return ToolResult(data=None, success=False, error=f"Not a file: {path}")

        # 二进制文件拒绝读取 — read_text(errors="replace") 产出乱码, 留在消息
        # 历史里会让 get_buffer_string MemoryError (v16 崩溃根因).
        if path.suffix.lower() in _BINARY_SUFFIXES:
            return ToolResult(
                data=None,
                success=False,
                error=(
                    f"{path.suffix} is a binary format, file_read_tool only reads text. "
                    "For PDFs: use search_tool to find the paper online, or grep_tool "
                    "to extract uncompressed text streams."
                ),
            )

        max_size = input_data.max_size_bytes or int(
            os.environ.get(
                "HUGINN_FILE_READ_MAX_SIZE_BYTES", str(DEFAULT_MAX_SIZE_BYTES)
            )
        )
        max_tokens = input_data.max_output_tokens or int(
            os.environ.get(
                "HUGINN_FILE_READ_MAX_OUTPUT_TOKENS", str(DEFAULT_MAX_OUTPUT_TOKENS)
            )
        )

        try:
            size = path.stat().st_size
            if size > max_size:
                return ToolResult(
                    data=None,
                    success=False,
                    error=(
                        f"File too large: {size} bytes (limit {max_size} bytes). "
                        "Use tail_lines or a smaller range."
                    ),
                )

            if input_data.tail_lines is not None and input_data.tail_lines > 0:
                selected, start = self._tail_lines(path, input_data.tail_lines)
                total = input_data.tail_lines
                end = start + len(selected)
            else:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
                total = len(lines)
                start = input_data.line_offset or 1
                end = (
                    total + 1
                    if input_data.n_lines is None
                    else start + input_data.n_lines
                )
                selected = lines[start - 1 : end - 1]

            selected, was_truncated = self._apply_token_cap(
                selected, start, max_tokens, path.suffix
            )
            numbered = "\n".join(
                f"{i + start:4d}  {line}" for i, line in enumerate(selected)
            )

            msg = f"Read lines {start}-{start + len(selected) - 1} of {total}."
            if was_truncated:
                msg += f" Truncated to stay under {max_tokens} tokens."

            return ToolResult(
                data={
                    "file_path": str(path),
                    "total_lines": total,
                    "start_line": start,
                    "content": numbered,
                    "message": msg,
                },
                success=True,
            )
        except Exception as e:
            return ToolResult(
                data=None, success=False, error=f"Failed to read file: {e}"
            )

    def _apply_token_cap(
        self,
        lines: list[str],
        start_line: int,
        max_tokens: int,
        suffix: str,
    ) -> tuple[list[str], bool]:
        """Truncate lines from the end until the output fits the token budget."""
        ext = suffix.lstrip(".").lower() or None
        selected = lines
        was_truncated = False
        while selected:
            text = "\n".join(
                f"{i + start_line:4d}  {line}" for i, line in enumerate(selected)
            )
            if rough_token_count_for_text(text, ext) <= max_tokens:
                break
            selected = selected[:-1]
            was_truncated = True
        return selected, was_truncated

    def _tail_lines(self, path: Path, n: int) -> tuple[list[str], int]:
        """Return the last n lines and the 1-based start line."""
        try:
            from huginn_ext import tail_lines  # type: ignore[import-not-found]

            lines = tail_lines(str(path), n)
            return lines, max(1, len(lines) - n + 1)
        except Exception:
            with open(path, encoding="utf-8", errors="replace") as f:
                all_lines = f.read().splitlines()
            start = max(1, len(all_lines) - n + 1)
            return all_lines[-n:], start
