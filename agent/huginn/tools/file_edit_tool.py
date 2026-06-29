"""File edit tool — precise string replacement in text files.

Used by the Coder agent to make minimal surgical edits. Always requires approval.
Supports dry-run diff preview and returns a unified diff in every result so the
LLM can verify what changed.
"""

from __future__ import annotations

import difflib
import hashlib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.permissions import PermissionChecker, PermissionConfig
from huginn.tools.base import HuginnTool
from huginn.types import PermissionMode, ToolContext, ToolResult


class FileEditToolInput(BaseModel):
    action: Literal["edit", "preview"] = Field(default="edit")
    file_path: str = Field(..., description="Path to file")
    old_string: str = Field(..., description="Exact text to replace")
    new_string: str = Field(..., description="Replacement text")
    working_dir: str | None = Field(default=None)
    dry_run: bool = Field(
        default=False,
        description="If True, return a diff preview without modifying the file.",
    )


def _make_diff(old: str, new: str, path: str) -> str:
    """Render a unified diff between ``old`` and ``new``."""
    diff_lines = difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"a/{Path(path).name}",
        tofile=f"b/{Path(path).name}",
    )
    return "".join(diff_lines)


def _content_hash(text: str) -> str:
    """Short SHA-256 hash for snapshot/rollback identification."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class FileEditTool(HuginnTool):
    """Edit text files by replacing a unique string.

    When ``dry_run`` is True (or ``action`` is ``"preview"``), the tool returns
    a unified diff without touching the file. On actual edits, the result
    includes both the diff and a ``snapshot_hash`` of the original content so
    callers can track what changed.
    """

    name = "file_edit_tool"
    category = "core"
    description = "Replace a unique substring in a text file with new content."
    destructive = True
    input_schema = FileEditToolInput

    async def call(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        input_data = FileEditToolInput(**args)
        work_dir = (
            Path(input_data.working_dir) if input_data.working_dir else Path.cwd()
        )
        path = work_dir / input_data.file_path
        if not path.is_absolute():
            path = path.resolve()

        # Prevent editing outside the working directory tree
        try:
            path.relative_to(work_dir.resolve())
        except ValueError:
            return ToolResult(
                data=None,
                success=False,
                error=f"Refusing to edit outside working directory: {path}",
            )

        if not path.exists():
            return ToolResult(data=None, success=False, error=f"File not found: {path}")

        # Refuse to read absurdly large files — protects against OOM
        _MAX_FILE_BYTES = 50 * 1024 * 1024  # 50 MB
        try:
            file_size = path.stat().st_size
            if file_size > _MAX_FILE_BYTES:
                return ToolResult(
                    data=None,
                    success=False,
                    error=f"File too large ({file_size} bytes, max {_MAX_FILE_BYTES})",
                )
        except OSError as e:
            return ToolResult(data=None, success=False, error=f"Cannot stat file: {e}")

        try:
            content = path.read_text(encoding="utf-8")
            if content.count(input_data.old_string) != 1:
                return ToolResult(
                    data=None,
                    success=False,
                    error=(
                        f"old_string occurs {content.count(input_data.old_string)} times; "
                        "must occur exactly once."
                    ),
                )
            new_content = content.replace(
                input_data.old_string, input_data.new_string, 1
            )
            diff = _make_diff(content, new_content, str(path))

            # Preview / dry-run: return the diff without writing.
            if input_data.dry_run or input_data.action == "preview":
                return ToolResult(
                    data={
                        "file_path": str(path),
                        "dry_run": True,
                        "diff": diff,
                        "message": f"Preview of edit to {path}.",
                    },
                    success=True,
                )

            # 真正写入前先过一遍权限检查. 如果是 ASK 模式, 只把 diff 预览返回去,
            # 不动文件 — 让 approval_callback 拿到 diff 后再决定要不要放行.
            perm_config = (
                context.config
                if context is not None and isinstance(context.config, PermissionConfig)
                else PermissionConfig()
            )
            checker = PermissionChecker(perm_config)
            perm_result = await checker.check(
                "file_edit_tool",
                is_destructive=True,
                args=args,
            )
            if perm_result.mode == PermissionMode.ASK:
                return ToolResult(
                    data={
                        "file_path": str(path),
                        "dry_run": True,
                        "needs_approval": True,
                        "diff": diff,
                        "reason": perm_result.reason,
                        "message": (
                            f"Edit to {path} requires approval — "
                            "diff preview generated, file unchanged."
                        ),
                    },
                    success=True,
                )
            if perm_result.mode == PermissionMode.DENY:
                return ToolResult(
                    data=None,
                    success=False,
                    error=perm_result.reason
                    or f"Edit to {path} denied by permission policy.",
                )

            path.write_text(new_content, encoding="utf-8")
            return ToolResult(
                data={
                    "file_path": str(path),
                    "replacements": 1,
                    "diff": diff,
                    "snapshot_hash": _content_hash(content),
                    "message": f"Edited {path}.",
                },
                success=True,
            )
        except Exception as e:
            return ToolResult(
                data=None, success=False, error=f"Failed to edit file: {e}"
            )
