"""LSP symbol-level editing tool — jedi-first, external langserver optional.

Upgrades the code agent's editing from "blind text replacement" to
"symbol-aware" operations: rename-with-references, find references, hover
signatures, syntax diagnostics, and code actions.

Design (see docs/lsp_hashline_spec.md, P0-1):
- **jedi-first**: pure-Python static analysis, no external process, sandbox-safe.
  Covers Python's rename / references / hover / definition / diagnostics.
- **graceful degradation**: jedi is imported lazily. If it's absent the tool
  returns an explicit error and the normal text-edit tools keep working
  (same philosophy as the Landlock sandbox fallback).
- **rename is the only destructive action**: it goes through the same
  PermissionChecker + work_dir boundary enforcement as the other edit tools.
  All other actions are read-only.
- **external langserver** (pyright/pygls) is a v2 upgrade path; the provider
  selection hooks exist here but python-only jedi is the working path.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.core_types import PermissionMode, ToolContext, ToolResult
from huginn.permissions import PermissionChecker, PermissionConfig
from huginn.tools.base import HuginnTool
from huginn.tools.file_edit_tool import _content_hash, _make_diff

logger = logging.getLogger(__name__)

_MAX_FILE_BYTES = 50 * 1024 * 1024  # 50 MB, same cap as file_edit_tool


class LspToolInput(BaseModel):
    action: Literal[
        "rename", "references", "hover", "diagnostics", "code_action", "definition"
    ] = Field(..., description="LSP 操作类型")
    file_path: str = Field(..., description="目标文件")
    line: int | None = Field(default=None, description="行号（0-based）")
    character: int | None = Field(default=None, description="列号（0-based）")
    name: str | None = Field(default=None, description="rename action 的新名称")
    working_dir: str | None = Field(default=None)
    provider: Literal["auto", "jedi", "external"] = Field(
        default="auto", description="auto=优先外部 langserver，无则回退 jedi"
    )


def _get_jedi():
    """Lazily import jedi. Returns the module or raises ImportError."""
    import jedi  # noqa: PLC0415 - lazy: graceful degradation when absent

    return jedi


def _resolve_path(file_path: str, work_dir: Path) -> Path:
    """Resolve a path and enforce it stays inside work_dir."""
    p = Path(file_path)
    if not p.is_absolute():
        p = work_dir / p
    p = p.resolve()
    p.relative_to(work_dir.resolve())  # boundary — may raise ValueError
    return p


def _read_file(path: Path) -> str | None:
    """Read a file, returning None (with a log) on any failure. Never raises."""
    try:
        if not path.exists():
            return None
        if path.stat().st_size > _MAX_FILE_BYTES:
            logger.warning("LSP skipping oversized file: %s", path)
            return None
        return path.read_text(encoding="utf-8")
    except OSError:
        logger.warning("LSP read failed: %s", path, exc_info=True)
        return None


class JediProvider:
    """Pure-Python symbol analysis backed by jedi."""

    def __init__(self, work_dir: Path) -> None:
        self.work_dir = work_dir
        self._jedi = _get_jedi()  # may raise ImportError -> caller degrades

    def _script(self, path: Path, code: str) -> Any:
        return self._jedi.Script(code, path=str(path))

    def _pos(
        self, code: str, line: int | None, character: int | None
    ) -> tuple[int, int] | None:
        """Convert 0-based (line, column) input to jedi's 1-based line + 0-based column.

        clamps column to the target line length so a slightly-out-of-range
        cursor still resolves to a nearby symbol instead of erroring.
        """
        if line is None:
            return None
        lines = code.splitlines()
        if not lines or line < 0 or line >= len(lines):
            return None
        line_len = len(lines[line])
        col = character if character is not None else 0
        col = max(0, min(col, line_len))
        return (line + 1, col)

    def definition(
        self, path: Path, code: str, line: int | None, character: int | None
    ) -> list[dict]:
        pos = self._pos(code, line, character)
        if pos is None:
            return []
        out = []
        for d in self._script(path, code).infer(*pos):
            out.append(
                {
                    "name": d.name,
                    "full_name": d.full_name or d.name,
                    "module": d.module_name,
                    "line": d.line - 1 if d.line else None,
                    "column": d.column if d.column is not None else None,
                    "path": d.module_path and str(d.module_path),
                    "description": d.description,
                }
            )
        return out

    def references(
        self, path: Path, code: str, line: int | None, character: int | None
    ) -> list[dict]:
        pos = self._pos(code, line, character)
        if pos is None:
            return []
        out = []
        for r in self._script(path, code).get_references(*pos, include_builtins=False):
            ref_path = r.module_path or r.module_name
            out.append(
                {
                    "name": r.name,
                    "path": str(ref_path) if isinstance(ref_path, Path) else ref_path,
                    "line": r.line - 1 if r.line else None,
                    "column": r.column if r.column is not None else None,
                }
            )
        return out

    def hover(
        self, path: Path, code: str, line: int | None, character: int | None
    ) -> dict:
        pos = self._pos(code, line, character)
        if pos is None:
            return {"found": False, "content": None}
        defs = self._script(path, code).infer(*pos)
        if not defs:
            return {"found": False, "content": None}
        d = defs[0]
        doc = (d.docstring() or "").strip()
        return {
            "found": True,
            "name": d.name,
            "full_name": d.full_name,
            "description": d.description,
            "docstring": doc,
        }

    def diagnostics(self, path: Path, code: str) -> list[dict]:
        out = []
        for e in self._script(path, code).get_syntax_errors():
            out.append(
                {
                    "severity": "error",
                    "message": e.get_message(),
                    "line": e.line - 1 if e.line else None,
                    "column": e.column if e.column is not None else None,
                }
            )
        return out

    def code_action(
        self, path: Path, code: str, line: int | None, character: int | None
    ) -> list[dict]:
        # jedi 只做静态分析, 不自动执行动作. 返回可用的符号级动作供 agent 决策.
        actions = ["definition", "references", "hover", "rename"]
        if line is not None and self.definition(path, code, line, character):
            actions.insert(0, "goto_definition")
        syntax = self.diagnostics(path, code)
        if syntax:
            actions.append("fix_syntax_errors")
        return [
            {"action": a, "execute": False, "note": "agent 需显式调用对应 action"}
            for a in actions
        ]

    def rename(
        self,
        path: Path,
        code: str,
        line: int | None,
        character: int | None,
        new_name: str,
    ) -> dict:
        """Symbol-level rename using jedi's refactoring engine.

        Returns changed files {abs_path: new_source} for targets inside the
        work_dir. Callers must gate this behind permissions and boundary checks.
        """
        pos = self._pos(code, line, character)
        if pos is None:
            raise ValueError("rename requires line/character to locate the symbol")
        # jedi 0.20: rename(self, line=None, column=None, *, new_name) — new_name 是
        # keyword-only, 必须显式命名, 不能走 *pos 展开 (会误判为第 4 个位置参数).
        refactoring = self._script(path, code).rename(
            line=pos[0], column=pos[1], new_name=new_name
        )
        changes: dict[str, str] = {}
        for target, changed_file in refactoring.get_changed_files().items():
            tp = Path(target)
            try:
                tp.relative_to(self.work_dir.resolve())
            except ValueError:
                logger.warning("rename skipping edit outside work_dir: %s", tp)
                continue
            # jedi 0.20: get_changed_files() 返回 ChangedFile 对象, 用 get_new_code()
            # 取新源码 (老版本直接返回字符串).
            new_code = (
                changed_file.get_new_code()
                if hasattr(changed_file, "get_new_code")
                else changed_file
            )
            changes[str(tp)] = new_code
        return changes


class LspTool(HuginnTool):
    """Symbol-level editing via LSP primitives (jedi-first)."""

    name = "lsp_tool"
    category = "core"
    description = (
        "Symbol-aware code operations: rename (with all references), find "
        "references, hover signatures, syntax diagnostics, and available code "
        "actions. Uses jedi for Python; degrades to an explicit error when "
        "unavailable. Only 'rename' modifies files."
    )
    destructive = False
    input_schema = LspToolInput

    def is_read_only(self, args: LspToolInput) -> bool:
        return args.action != "rename"

    def is_destructive(self, args: LspToolInput) -> bool:
        return args.action == "rename"

    def _provider(self, work_dir: Path):
        # provider='external' 或 'auto' 时优先尝试外部 langserver; 当前仅 jedi
        # 是可用实现, 外部 langserver 留作 v2 升级路径 (见 spec §3.4).
        return JediProvider(work_dir)

    async def _execute(self, args: LspToolInput, context: ToolContext) -> ToolResult:
        if not isinstance(args, LspToolInput):
            args = LspToolInput(**args)
        work_dir = Path(args.working_dir) if args.working_dir else Path.cwd()
        try:
            path = _resolve_path(args.file_path, work_dir)
        except ValueError:
            return ToolResult(
                data=None,
                success=False,
                error=f"Refusing to access outside working directory: {args.file_path}",
            )

        code = _read_file(path)
        if code is None:
            return ToolResult(
                data=None,
                success=False,
                error=f"Unable to read file: {path}",
            )

        try:
            provider = self._provider(work_dir)
        except ImportError as exc:
            return ToolResult(
                data=None,
                success=False,
                error=(
                    f"LSP unavailable: {exc}. Install jedi to enable symbol-level "
                    "editing; text-based file_edit_tool still works."
                ),
            )

        try:
            if args.action in {
                "references",
                "hover",
                "definition",
                "diagnostics",
                "code_action",
            }:
                return await self._run_read_action(provider, path, code, args)
            return await self._run_rename(provider, path, code, args, context, work_dir)
        except Exception as exc:
            return ToolResult(
                data=None,
                success=False,
                error=f"LSP {args.action} failed: {type(exc).__name__}: {exc}",
            )

    async def _run_read_action(
        self, provider: JediProvider, path: Path, code: str, args: LspToolInput
    ) -> ToolResult:
        if args.action == "definition":
            data = provider.definition(path, code, args.line, args.character)
        elif args.action == "references":
            data = provider.references(path, code, args.line, args.character)
        elif args.action == "hover":
            data = provider.hover(path, code, args.line, args.character)
        elif args.action == "diagnostics":
            data = provider.diagnostics(path, code)
        else:  # code_action
            data = provider.code_action(path, code, args.line, args.character)
        return ToolResult(
            data={"action": args.action, "file_path": str(path), "result": data},
            success=True,
        )

    async def _run_rename(
        self,
        provider: JediProvider,
        path: Path,
        code: str,
        args: LspToolInput,
        context: ToolContext,
        work_dir: Path,
    ) -> ToolResult:
        if not args.name:
            return ToolResult(
                data=None, success=False, error="rename requires a 'name'"
            )

        # 先算 diff 供 ASK 预览, 再写.
        changes = provider.rename(path, code, args.line, args.character, args.name)
        if not changes:
            return ToolResult(
                data=None,
                success=False,
                error="No changes produced by rename (symbol not found at position?).",
            )

        # 生成统一 diff 摘要供确认
        diffs: dict[str, str] = {}
        for fp, new_code in changes.items():
            old = _read_file(Path(fp))
            diffs[fp] = _make_diff(old or "", new_code, fp)

        perm_config = (
            context.config
            if context is not None and isinstance(context.config, PermissionConfig)
            else PermissionConfig()
        )
        checker = PermissionChecker(perm_config)
        perm_result = await checker.check(
            "lsp_tool",
            is_destructive=True,
            args={"action": "rename", "file_path": str(path), "name": args.name},
        )
        if perm_result.mode == PermissionMode.ASK:
            return ToolResult(
                data={
                    "action": "rename",
                    "dry_run": True,
                    "needs_approval": True,
                    "diffs": diffs,
                    "reason": perm_result.reason,
                    "message": f"Rename to '{args.name}' requires approval — files unchanged.",
                },
                success=True,
            )
        if perm_result.mode == PermissionMode.DENY:
            return ToolResult(
                data=None,
                success=False,
                error=perm_result.reason or "Rename denied by permission policy.",
            )

        # 落盘: 记录每个文件编辑前 hash 供 Hashline 闭环.
        written: list[dict] = []
        for fp, new_code in changes.items():
            target = Path(fp)
            before = _read_file(target) or ""
            target.write_text(new_code, encoding="utf-8")
            written.append(
                {
                    "file_path": fp,
                    "snapshot_hash": _content_hash(before),
                }
            )
        return ToolResult(
            data={
                "action": "rename",
                "name": args.name,
                "files": written,
                "diffs": diffs,
                "message": f"Renamed symbol to '{args.name}' across {len(written)} file(s).",
            },
            success=True,
        )
