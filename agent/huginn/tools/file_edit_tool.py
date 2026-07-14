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


# OAK 启发: 结构文件语义 diff — 对 CIF/POSCAR/INCAR 做语义级变更说明
# 而非行级 unified diff. 复用 pymatgen 解析结构, 提取人类可读的变更.
_STRUCTURE_EXTS = {".cif", ".poscar", ".vasp", ".contcar"}
_PARAM_EXTS = {".incar", ".yaml", ".yml", ".json"}


def _semantic_diff(old: str, new: str, path: str) -> str | None:
    """对结构/参数文件返回语义级 diff 说明, 不支持的格式返回 None."""
    ext = Path(path).suffix.lower()
    if ext in _STRUCTURE_EXTS:
        try:
            from pymatgen.core import Structure
            s_old = Structure.from_str(old, fmt="cif" if ext == ".cif" else "poscar")
            s_new = Structure.from_str(new, fmt="cif" if ext == ".cif" else "poscar")
            changes = []
            if s_old.composition.reduced_formula != s_new.composition.reduced_formula:
                changes.append(f"组分: {s_old.composition.reduced_formula} → {s_new.composition.reduced_formula}")
            if abs(s_old.lattice.volume - s_new.lattice.volume) > 0.01:
                changes.append(f"体积: {s_old.lattice.volume:.2f} → {s_new.lattice.volume:.2f} Å³")
            if abs(s_old.lattice.a - s_new.lattice.a) > 0.001:
                changes.append(f"a: {s_old.lattice.a:.3f} → {s_new.lattice.a:.3f} Å")
            if abs(s_old.lattice.b - s_new.lattice.b) > 0.001:
                changes.append(f"b: {s_old.lattice.b:.3f} → {s_new.lattice.b:.3f} Å")
            if abs(s_old.lattice.c - s_new.lattice.c) > 0.001:
                changes.append(f"c: {s_old.lattice.c:.3f} → {s_new.lattice.c:.3f} Å")
            if len(s_old) != len(s_new):
                changes.append(f"原子数: {len(s_old)} → {len(s_new)}")
            return "; ".join(changes) if changes else None
        except Exception:
            return None
    if ext in _PARAM_EXTS:
        try:
            import json
            old_d = json.loads(old) if ext == ".json" else _parse_kv(old)
            new_d = json.loads(new) if ext == ".json" else _parse_kv(new)
            changes = []
            for k in sorted(set(old_d) | set(new_d)):
                ov, nv = old_d.get(k), new_d.get(k)
                if ov != nv:
                    changes.append(f"{k}: {ov} → {nv}")
            return "; ".join(changes) if changes else None
        except Exception:
            return None
    return None


def _parse_kv(text: str) -> dict:
    """解析 INCAR 风格的 key = value 格式."""
    result = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("!"):
            continue
        if "=" in line:
            k, _, v = line.partition("=")
            result[k.strip().upper()] = v.strip()
    return result


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
            # OAK 启发: 结构文件附加语义 diff (组分/体积/参数变化)
            sem_diff = _semantic_diff(content, new_content, str(path))

            # Preview / dry-run: return the diff without writing.
            if input_data.dry_run or input_data.action == "preview":
                return ToolResult(
                    data={
                        "file_path": str(path),
                        "dry_run": True,
                        "diff": diff,
                        "semantic_diff": sem_diff,
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
                        "semantic_diff": sem_diff,
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
                    "semantic_diff": sem_diff,
                    "snapshot_hash": _content_hash(content),
                    "message": f"Edited {path}.",
                },
                success=True,
            )
        except Exception as e:
            return ToolResult(
                data=None, success=False, error=f"Failed to edit file: {e}"
            )
