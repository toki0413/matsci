"""Multi-file atomic edit tool — coordinated edits across several files at once.

Mirrors Claude Code's multi-file coordinated edit: one tool call applies a batch
of string replacements across multiple files. When ``atomic`` is True (default)
the tool validates every edit up front and refuses to write anything if any
single edit would fail, so a cross-file refactor either lands completely or not
at all.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.core_types import PermissionMode, ToolContext, ToolResult
from huginn.permissions import PermissionChecker, PermissionConfig
from huginn.tools.base import HuginnTool
from huginn.tools.file_edit_tool import _content_hash, _make_diff
from huginn.tools.profile import ToolProfile

logger = logging.getLogger(__name__)


class SingleEdit(BaseModel):
    """One replacement to apply inside a single file."""

    file_path: str = Field(..., description="Path to the file to edit (relative to working_dir or absolute).")
    old_string: str = Field(..., description="Exact text to replace; must occur exactly once in the file.")
    new_string: str = Field(..., description="Replacement text.")
    expected_hash: str | None = Field(
        default=None,
        description="Expected SHA-256 (first 16 hex chars) of this file's current content, "
        "from a prior snapshot_hash. If provided and the on-disk content differs, the "
        "whole atomic batch is refused (unless hash_policy='warn').",
    )
    hash_policy: Literal["strict", "warn", "off"] = Field(
        default="strict",
        description="strict: refuse the batch when expected_hash mismatches; "
        "warn: log the mismatch but continue; off: skip hash check.",
    )


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


# ── 校验阶段 helpers ─────────────────────────────────────────────────────


def _me_failure(error: str, data: dict[str, Any] | None = None) -> ToolResult:
    """构造失败结果 (data 默认 None, 保持原语义)."""
    return ToolResult(data=data, success=False, error=error)


def _me_group_edits(edits: list[SingleEdit]) -> OrderedDict[str, list[SingleEdit]]:
    """按 file_path 分组并保持首次出现顺序."""
    grouped: OrderedDict[str, list[SingleEdit]] = OrderedDict()
    for edit in edits:
        grouped.setdefault(edit.file_path, []).append(edit)
    return grouped


def _me_load_file(
    file_path: str, work_dir: Path
) -> tuple[Path | None, str | None, ToolResult | None]:
    """解析路径、做边界/存在/大小检查并读取 UTF-8 内容."""
    try:
        path = _resolve_path(file_path, work_dir)
    except ValueError:
        return (
            None,
            None,
            _me_failure(
                "Atomic batch aborted — refusing to edit outside working directory: "
                f"{file_path}"
            ),
        )
    if not path.exists():
        return None, None, _me_failure(f"Atomic batch aborted — file not found: {path}")
    try:
        size = path.stat().st_size
    except OSError as e:
        return None, None, _me_failure(f"Atomic batch aborted — cannot stat file: {e}")
    if size > _MAX_FILE_BYTES:
        return (
            None,
            None,
            _me_failure(
                f"Atomic batch aborted — file too large ({size} bytes, "
                f"max {_MAX_FILE_BYTES}): {path}"
            ),
        )
    try:
        content = path.read_text(encoding="utf-8")
    except Exception as e:
        return None, None, _me_failure(f"Atomic batch aborted — cannot read {path}: {e}")
    return path, content, None


def _me_check_snapshot_hash(
    edits: list[SingleEdit], content: str, path: Path
) -> ToolResult | None:
    """按首个携带 expected_hash 的 edit 校验一次内容 hash."""
    for sedit in edits:
        if sedit.expected_hash and sedit.hash_policy != "off":
            current_hash = _content_hash(content)
            if current_hash != sedit.expected_hash:
                msg = (
                    f"File changed since snapshot (expected {sedit.expected_hash}, "
                    f"got {current_hash}): {path}. Refusing atomic batch."
                )
                if sedit.hash_policy == "strict":
                    return _me_failure(msg, {"current_hash": current_hash})
                logger.warning(msg)
            break  # 只按首个带 hash 的 edit 校验一次
    return None


def _me_apply_edits(
    content: str, edits: list[SingleEdit], path: Path
) -> tuple[str | None, ToolResult | None]:
    """链式应用单文件全部 edit; 每个 old_string 在当前内容中必须唯一."""
    for edit in edits:
        occ = content.count(edit.old_string)
        if occ != 1:
            return (
                None,
                _me_failure(
                    f"Atomic batch aborted — old_string occurs {occ} times "
                    f"in {path}; must occur exactly once."
                ),
            )
        content = content.replace(edit.old_string, edit.new_string, 1)
    return content, None


def _me_validate_file(
    file_path: str, edits: list[SingleEdit], work_dir: Path
) -> tuple[tuple[Path, str, str, str, int] | None, ToolResult | None]:
    """单文件预检: 读取 → hash → 链式替换 → diff."""
    path, content, failure = _me_load_file(file_path, work_dir)
    if failure is not None:
        return None, failure
    assert path is not None and content is not None
    original = content
    failure = _me_check_snapshot_hash(edits, content, path)
    if failure is not None:
        return None, failure
    new_content, failure = _me_apply_edits(content, edits, path)
    if failure is not None:
        return None, failure
    assert new_content is not None
    diff = _make_diff(original, new_content, str(path))
    return (path, original, new_content, diff, len(edits)), None


def _me_validate_batch(
    edits: list[SingleEdit], work_dir: Path
) -> tuple[list[tuple[Path, str, str, str, int]], ToolResult | None]:
    """按文件分组顺序预检; 任一文件失败则整批中止."""
    validated: list[tuple[Path, str, str, str, int]] = []
    for file_path, file_edits in _me_group_edits(edits).items():
        entry, failure = _me_validate_file(file_path, file_edits, work_dir)
        if failure is not None:
            return [], failure
        assert entry is not None
        validated.append(entry)
    return validated, None


# ── 结果汇总 helpers ─────────────────────────────────────────────────────


def _me_edit_diffs(validated: list[tuple[Path, str, str, str, int]]) -> list[dict[str, Any]]:
    """把预检结果投影为 ``[{file_path, diff}]``."""
    return [{"file_path": str(path), "diff": diff} for (path, _, _, diff, _) in validated]


def _me_preview_result(
    validated: list[tuple[Path, str, str, str, int]], n_edits: int
) -> ToolResult:
    """dry_run / preview 分支: 只返回合并 diff."""
    return ToolResult(
        data={
            "dry_run": True,
            "edits": _me_edit_diffs(validated),
            "message": f"Preview of {n_edits} edits across {len(validated)} files.",
        },
        success=True,
    )


def _me_approval_result(
    validated: list[tuple[Path, str, str, str, int]], reason: str | None
) -> ToolResult:
    """ASK 模式: 返回 diff 预览并要求审批."""
    return ToolResult(
        data={
            "dry_run": True,
            "needs_approval": True,
            "edits": _me_edit_diffs(validated),
            "reason": reason,
            "message": "Batch edit requires approval — diff previews generated, no files changed.",
        },
        success=True,
    )


async def _me_check_permission(context: ToolContext | None, args: dict[str, Any]) -> Any:
    """执行权限检查."""
    perm_config = (
        context.config
        if context is not None and isinstance(context.config, PermissionConfig)
        else PermissionConfig()
    )
    checker = PermissionChecker(perm_config)
    return await checker.check("multi_edit_tool", is_destructive=True, args=args)


def _me_write_and_report(
    validated: list[tuple[Path, str, str, str, int]], n_edits: int
) -> ToolResult:
    """顺序写入全部文件并汇总; 写入阶段异常返回部分成功语义."""
    applied: list[dict[str, Any]] = []
    try:
        for (path, original, new_content, diff, count) in validated:
            path.write_text(new_content, encoding="utf-8")
            applied.append(
                {
                    "file_path": str(path),
                    "replacements": count,
                    "diff": diff,
                    "snapshot_hash": _content_hash(original),
                }
            )
    except Exception as e:
        return ToolResult(
            data={
                "applied": applied,
                "files_changed": len(applied),
                "partial_failure": True,
            },
            success=False,
            error=(
                f"Batch edit failed mid-write after {len(applied)}/"
                f"{len(validated)} files: {e}"
            ),
        )
    return ToolResult(
        data={
            "applied": applied,
            "files_changed": len(applied),
            "message": f"Applied {n_edits} edits across {len(applied)} files.",
        },
        success=True,
    )


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
            Path(input_data.working_dir) if input_data.working_dir else self.default_work_dir()
        )

        # 1. 按文件分组顺序预检 (原子性: 任一失败整批中止)
        validated, failure = _me_validate_batch(input_data.edits, work_dir)
        if failure is not None:
            return failure

        # 2. dry_run / preview: 只返回合并 diff, 不落盘
        if input_data.dry_run or input_data.action == "preview":
            return _me_preview_result(validated, len(input_data.edits))

        # 3. 权限检查: ASK 模式只给预览, DENY 直接拒
        perm_result = await _me_check_permission(context, args)
        if perm_result.mode == PermissionMode.ASK:
            return _me_approval_result(validated, perm_result.reason)
        if perm_result.mode == PermissionMode.DENY:
            return _me_failure(perm_result.reason or "Batch edit denied by permission policy.")

        # 4. 顺序写入 (预检已通过)
        return _me_write_and_report(validated, len(input_data.edits))
