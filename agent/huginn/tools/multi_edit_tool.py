"""Multi-file atomic edit tool — coordinated edits across several files at once.

Mirrors Claude Code's multi-file coordinated edit: one tool call applies a batch
of string replacements across multiple files. When ``atomic`` is True (default)
the tool validates every edit up front and refuses to write anything if any
single edit would fail, so a cross-file refactor either lands completely or not
at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.permissions import PermissionChecker, PermissionConfig
from huginn.tools.base import HuginnTool
from huginn.tools.file_edit_tool import _content_hash, _make_diff
from huginn.tools.profile import ToolProfile
from huginn.types import PermissionMode, ToolContext, ToolResult


class SingleEdit(BaseModel):
    """One replacement to apply inside a single file."""

    file_path: str = Field(..., description="Path to the file to edit (relative to working_dir or absolute).")
    old_string: str = Field(..., description="Exact text to replace; must occur exactly once in the file.")
    new_string: str = Field(..., description="Replacement text.")


class MultiEditToolInput(BaseModel):
    action: Literal["edit", "preview"] = Field(default="edit")
    edits: list[SingleEdit] = Field(
        ...,
        description="List of edits to apply. Each edit targets one file; multiple edits may target the same file.",
        min_length=1,
    )
    working_dir: str | None = Field(default=None)
    atomic: bool = Field(
        default=True,
        description="If True, validate all edits first and write nothing if any would fail.",
    )
    dry_run: bool = Field(
        default=False,
        description="If True, return a combined diff preview without modifying any file.",
    )


_MAX_FILE_BYTES = 50 * 1024 * 1024  # 50 MB, same cap as file_edit_tool


def _resolve_path(file_path: str, work_dir: Path) -> Path:
    """Resolve a path and enforce it stays inside work_dir."""
    p = Path(file_path)
    if not p.is_absolute():
        p = work_dir / p
    p = p.resolve()
    # Boundary check — refuse to touch anything outside the workspace tree.
    p.relative_to(work_dir.resolve())
    return p


class MultiEditTool(HuginnTool):
    """Apply a batch of string replacements across multiple files atomically."""

    name = "multi_edit_tool"
    category = "core"
    description = "Apply multiple string replacements across several files in one call. Atomic by default — validates all edits first, writes nothing if any would fail."
    destructive = True
    input_schema = MultiEditToolInput
    # light tier: 同 file_edit_tool, CPU 开销可忽略, 走 light 信号量
    profile = ToolProfile(cost_tier="light")

    async def call(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        input_data = MultiEditToolInput(**args)
        work_dir = (
            Path(input_data.working_dir) if input_data.working_dir else Path.cwd()
        )

        # ── 1. 按文件分组 + 链式预检 ──
        # 同一文件的多个 edit 必须顺序应用: edit2 在 edit1 应用后的内容上找 old_string.
        # 否则 edit2 的 new_content 会基于原始内容算出, 写入时覆盖 edit1.
        from collections import OrderedDict

        grouped: "OrderedDict[str, list[SingleEdit]]" = OrderedDict()
        for edit in input_data.edits:
            grouped.setdefault(edit.file_path, []).append(edit)

        validated: list[tuple[Path, str, str, str, int]] = []
        # tuple = (path, original_content, final_content, diff, n_edits)
        for file_path, edits in grouped.items():
            # 解析路径 + 边界检查 (只做一次)
            try:
                path = _resolve_path(file_path, work_dir)
            except ValueError:
                return ToolResult(
                    data=None,
                    success=False,
                    error=f"Atomic batch aborted — refusing to edit outside working directory: {file_path}",
                )
            if not path.exists():
                return ToolResult(
                    data=None,
                    success=False,
                    error=f"Atomic batch aborted — file not found: {path}",
                )
            try:
                size = path.stat().st_size
                if size > _MAX_FILE_BYTES:
                    return ToolResult(
                        data=None,
                        success=False,
                        error=f"Atomic batch aborted — file too large ({size} bytes, max {_MAX_FILE_BYTES}): {path}",
                    )
            except OSError as e:
                return ToolResult(
                    data=None,
                    success=False,
                    error=f"Atomic batch aborted — cannot stat file: {e}",
                )
            try:
                content = path.read_text(encoding="utf-8")
            except Exception as e:
                return ToolResult(
                    data=None,
                    success=False,
                    error=f"Atomic batch aborted — cannot read {path}: {e}",
                )

            original = content
            # 链式应用: 每个 edit 在当前 content 上找 old_string, 必须唯一
            for edit in edits:
                occ = content.count(edit.old_string)
                if occ != 1:
                    return ToolResult(
                        data=None,
                        success=False,
                        error=(
                            f"Atomic batch aborted — old_string occurs {occ} times "
                            f"in {path}; must occur exactly once."
                        ),
                    )
                content = content.replace(edit.old_string, edit.new_string, 1)
            diff = _make_diff(original, content, str(path))
            validated.append((path, original, content, diff, len(edits)))

        # ── 2. dry_run / preview: 只返回合并 diff, 不落盘 ──
        if input_data.dry_run or input_data.action == "preview":
            return ToolResult(
                data={
                    "dry_run": True,
                    "edits": [
                        {"file_path": str(path), "diff": diff}
                        for (path, _, _, diff, _) in validated
                    ],
                    "message": (
                        f"Preview of {len(input_data.edits)} edits "
                        f"across {len(validated)} files."
                    ),
                },
                success=True,
            )

        # ── 3. 权限检查: ASK 模式只给预览, DENY 直接拒 ──
        perm_config = (
            context.config
            if context is not None and isinstance(context.config, PermissionConfig)
            else PermissionConfig()
        )
        checker = PermissionChecker(perm_config)
        perm_result = await checker.check(
            "multi_edit_tool",
            is_destructive=True,
            args=args,
        )
        if perm_result.mode == PermissionMode.ASK:
            return ToolResult(
                data={
                    "dry_run": True,
                    "needs_approval": True,
                    "edits": [
                        {"file_path": str(path), "diff": diff}
                        for (path, _, _, diff, _) in validated
                    ],
                    "reason": perm_result.reason,
                    "message": "Batch edit requires approval — diff previews generated, no files changed.",
                },
                success=True,
            )
        if perm_result.mode == PermissionMode.DENY:
            return ToolResult(
                data=None,
                success=False,
                error=perm_result.reason
                or "Batch edit denied by permission policy.",
            )

        # ── 4. 顺序写入 (atomic 已预检通过, 写入阶段失败属异常) ──
        applied: list[dict[str, Any]] = []
        try:
            for (path, original, new_content, diff, n_edits) in validated:
                path.write_text(new_content, encoding="utf-8")
                applied.append(
                    {
                        "file_path": str(path),
                        "replacements": n_edits,
                        "diff": diff,
                        "snapshot_hash": _content_hash(original),
                    }
                )
            return ToolResult(
                data={
                    "applied": applied,
                    "files_changed": len(applied),
                    "message": (
                        f"Applied {len(input_data.edits)} edits "
                        f"across {len(applied)} files."
                    ),
                },
                success=True,
            )
        except Exception as e:
            # 写入阶段失败: 已写入的文件无法自动回滚 (snapshot_hash 留了恢复依据).
            return ToolResult(
                data={
                    "applied": applied,
                    "files_changed": len(applied),
                    "partial_failure": True,
                },
                success=False,
                error=f"Batch edit failed mid-write after {len(applied)}/{len(validated)} files: {e}",
            )
