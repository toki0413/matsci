"""File read tool — read text files with optional line range.

A read-only tool for inspecting source code, logs, and configuration files.
Safe to auto-execute.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.core_types import ToolContext, ToolResult
from huginn.tools.base import HuginnTool
from huginn.utils.tokens import rough_token_count_for_text

DEFAULT_MAX_SIZE_BYTES = 256 * 1024
DEFAULT_MAX_OUTPUT_TOKENS = 25000


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
    window_offset: int | None = Field(
        default=None,
        ge=0,
        description=(
            "0-based byte offset for a lazy window read. 设置后按字节 seek "
            "只读一小块(见 window_size), 不整体载入超大文件, 也跳过 256KB 整读上限."
        ),
    )
    window_size: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Max bytes to read in window mode (默认取 max_size_bytes). "
            "仅与 window_offset 搭配生效."
        ),
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

        # PDF 走 KB 的 pymupdf + OCR 链, 不走 read_text(errors="replace").
        # 旧版 read_text 产 256KB 乱码, 留在消息历史里 get_buffer_string
        # MemoryError (v16 崩溃根因). KB 的 _extract_text 用 pymupdf 提取
        # 真正文本, 扫描件 fallback 到 OCR.
        if path.suffix.lower() == ".pdf":
            try:
                from huginn.knowledge.store import _extract_text

                content = path.read_bytes()
                # pymupdf 流式提取, 不受 max_size 限制 — 输出靠 max_tokens 截断.
                # PDF 原始 2.3MB 但提取后文本通常 <100KB.
                pdf_text = _extract_text(str(path), content)
                if pdf_text:
                    lines = pdf_text.splitlines()
                    total = len(lines)
                    # 支持 line_offset/n_lines 读 PDF 中间段落 — 之前固定从第 1 行
                    # 截断, agent 读不到 Methods 中间部分, 只能反复 fetch 外部论文
                    # (网络不稳定时崩溃). 现在 agent 可分页读完整 PDF.
                    start = input_data.line_offset or 1
                    end = (
                        total + 1
                        if input_data.n_lines is None
                        else start + input_data.n_lines
                    )
                    selected = lines[start - 1 : end - 1]
                    selected, was_truncated = self._apply_token_cap(
                        selected, start, max_tokens, ".pdf"
                    )
                    numbered = "\n".join(
                        f"{i + start:4d}  {line}" for i, line in enumerate(selected)
                    )
                    msg = f"Read PDF ({len(pdf_text)} chars extracted, lines {start}-{start + len(selected) - 1} of {total})."
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
                    data=None,
                    success=False,
                    error=f"PDF extraction failed: {e}",
                )

        try:
            size = path.stat().st_size
            window_mode = input_data.window_offset is not None
            # 整读保护: 仅非窗口模式才拦超大文件. 窗口模式按字节 seek 只读
            # 一小块(< window_size), 即便文件极大也内存受控, 因此跳过该上限.
            if not window_mode and size > max_size:
                return ToolResult(
                    data=None,
                    success=False,
                    error=(
                        f"File too large: {size} bytes (limit {max_size} bytes). "
                        "Use tail_lines, a smaller line range, or window_offset "
                        "/window_size for a lazy window read."
                    ),
                )

            if window_mode:
                # 懒读/分块窗口: 按字节 seek 读一块, 不整体载入超大文件.
                window_bytes = input_data.window_size or max_size
                selected, start = self._read_window(
                    path, input_data.window_offset or 0, window_bytes, size
                )
                total = self._count_total_lines(path)
                if start is None:
                    return ToolResult(
                        data=None,
                        success=False,
                        error=(
                            f"Window offset {input_data.window_offset} is beyond "
                            f"end of file ({size} bytes)."
                        ),
                    )
            elif input_data.tail_lines is not None and input_data.tail_lines > 0:
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

            if window_mode:
                msg = (
                    f"Read window (offset={input_data.window_offset} bytes) lines "
                    f"{start}-{start + len(selected) - 1} of {total}."
                )
            else:
                msg = f"Read lines {start}-{start + len(selected) - 1} of {total}."
            if was_truncated:
                msg += f" Truncated to stay under {max_tokens} tokens."

            data = {
                "file_path": str(path),
                "total_lines": total,
                "start_line": start,
                "content": numbered,
                "message": msg,
            }
            if window_mode:
                data["window_offset"] = input_data.window_offset or 0
                data["window_size"] = input_data.window_size or max_size
            return ToolResult(data=data, success=True)
        except Exception as e:
            return ToolResult(
                data=None, success=False, error=f"Failed to read file: {e}"
            )

    def _read_window(
        self, path: Path, byte_offset: int, byte_size_limit: int, file_size: int
    ) -> tuple[list[str], int | None]:
        """按字节 seek 懒读一块文本窗口并切行 (不整体载入文件).

        逐字读取超大文件时把整文件留盘上, 只把 [byte_offset,
        byte_offset + byte_size_limit) 这一小片读进内存. 返回 (选中行列表,
        该窗口首行的 1-based 行号); 偏移越过文件末尾时行号返回 None. 首行
        行号通过流式统计 offset 之前的新行数(常量内存)得出.

        Returns:
            tuple[list[str], int | None]: (窗口内切出的行, 起始 1-based 行号).
        """
        if byte_offset >= file_size:
            return [], None
        chunk_bytes = min(byte_size_limit, max(1, file_size - byte_offset))
        with open(path, "rb") as f:
            f.seek(byte_offset)
            data = f.read(chunk_bytes)
        text = data.decode("utf-8", errors="replace")
        start = 1
        if byte_offset:
            start = self._count_lines_before(path, byte_offset) + 1
        return text.splitlines(), start

    @staticmethod
    def _count_lines_before(path: Path, upto_bytes: int) -> int:
        """流式统计 [0, upto_bytes) 字节区间内的换行数 (常量内存)."""
        count = 0
        remaining = upto_bytes
        with open(path, "rb") as f:
            while remaining > 0:
                chunk = f.read(min(1 << 20, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                count += chunk.count(b"\n")
        return count

    @staticmethod
    def _count_total_lines(path: Path) -> int:
        """流式统计全文件换行数 (常量内存), 供窗口模式报告 total_lines."""
        count = 0
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                count += chunk.count(b"\n")
        return count

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
