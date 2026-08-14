"""Git tool — git introspection + write actions for the Coder agent.

Read actions (status/diff/log) are safe and auto-approved.
Write actions (add/commit/branch_create/checkout) mutate the repo and are
gated by the permission system (destructive=True).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.core_types import PermissionMode, ToolContext, ToolResult
from huginn.permissions import PermissionChecker, PermissionConfig
from huginn.security.revertible import register_compensator
from huginn.tools.base import HuginnTool
from huginn.tools.profile import ToolProfile


def _git(work_dir: str, *args: str) -> str:
    """Run a git command in work_dir, return stripped stdout (or '' on failure)."""
    try:
        r = subprocess.run(
            ["git", *args],
            cwd=work_dir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30.0,
        )
        return (r.stdout or "").strip() if r.returncode == 0 else ""
    except Exception:
        return ""


# ── 出站写操作的补偿逆 (RevertibleEffect / Cordis 时间可组合性) ────
# git 写操作成功后, git_tool 把补偿逆登记进 RevertibleContext; 上层回滚时
# 按注册的补偿器执行 (commit→reset --soft, checkout→回原引用, add→reset).
def _register_git_compensators() -> None:
    register_compensator("git_commit", _compensate_commit)
    register_compensator("git_checkout", _compensate_checkout)
    register_compensator("git_add", _compensate_add)


def _compensate_commit(payload: dict[str, Any]) -> None:
    """撤销一次 commit: reset --soft 回退 HEAD, 保留工作区改动."""
    work_dir = payload.get("work_dir")
    prev_hash = payload.get("prev_hash")
    if work_dir and prev_hash:
        _git(work_dir, "reset", "--soft", str(prev_hash))


def _compensate_checkout(payload: dict[str, Any]) -> None:
    """回到切换前的分支/引用 (branch_create/checkout 的逆)."""
    work_dir = payload.get("work_dir")
    prev_ref = payload.get("prev_ref")
    if work_dir and prev_ref:
        _git(work_dir, "checkout", str(prev_ref))


def _compensate_add(payload: dict[str, Any]) -> None:
    """撤销 git add 的暂存, 保留文件改动."""
    work_dir = payload.get("work_dir")
    if work_dir:
        _git(work_dir, "reset")


_register_git_compensators()


class GitToolInput(BaseModel):
    action: Literal["status", "diff", "log", "add", "commit", "branch_create", "checkout"] = Field(default="status")
    working_dir: str | None = Field(default=None)
    max_lines: int = Field(default=200, ge=1)
    # write-action params
    paths: list[str] | None = Field(default=None, description="Files to stage (action=add). None = stage all.")
    message: str | None = Field(default=None, description="Commit message (action=commit).")
    branch: str | None = Field(default=None, description="Branch name (action=branch_create/checkout).")


_READ_ACTIONS = {"status", "diff", "log"}
_WRITE_ACTIONS = {"add", "commit", "branch_create", "checkout"}


class GitTool(HuginnTool):
    """Git introspection + write actions."""

    name = "git_tool"
    category = "core"
    description = "Run git commands: read (status/diff/log) and write (add/commit/branch_create/checkout)."
    input_schema = GitToolInput
    # 读多写少, 默认 light; 写动作在 call() 里过权限检查
    profile = ToolProfile(cost_tier="light")

    def is_read_only(self, args: GitToolInput) -> bool:
        return args.action in _READ_ACTIONS

    def is_destructive(self, args: GitToolInput) -> bool:
        return args.action in _WRITE_ACTIONS

    async def call(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        input_data = GitToolInput(**args)
        work_dir = (
            Path(input_data.working_dir) if input_data.working_dir else Path.cwd()
        )

        # ── 构造命令 ──
        cmd = _build_cmd(input_data)
        if cmd is None:
            return ToolResult(
                data=None,
                success=False,
                error=f"Unknown or incomplete git action: {input_data.action}",
            )

        # ── 写动作过权限 ──
        if input_data.action in _WRITE_ACTIONS:
            perm_config = (
                context.config
                if context is not None and isinstance(context.config, PermissionConfig)
                else PermissionConfig()
            )
            checker = PermissionChecker(perm_config)
            perm_result = await checker.check(
                "git_tool",
                is_destructive=True,
                args=args,
            )
            if perm_result.mode == PermissionMode.ASK:
                return ToolResult(
                    data={
                        "dry_run": True,
                        "needs_approval": True,
                        "action": input_data.action,
                        "command": " ".join(cmd),
                        "reason": perm_result.reason,
                        "message": f"Git {input_data.action} requires approval — no changes made.",
                    },
                    success=True,
                )
            if perm_result.mode == PermissionMode.DENY:
                return ToolResult(
                    data=None,
                    success=False,
                    error=perm_result.reason or f"Git {input_data.action} denied by permission policy.",
                )

        # ── 写动作执行前捕获前状态 (供补偿逆撤销) ──
        prev_hash: str | None = None
        prev_ref: str | None = None
        if input_data.action in ("commit", "add", "branch_create", "checkout"):
            prev_hash = _git(str(work_dir), "rev-parse", "HEAD") or None
            prev_ref = _git(str(work_dir), "rev-parse", "--abbrev-ref", "HEAD")
            if prev_ref == "HEAD":
                prev_ref = prev_hash  # detached HEAD → 回到该 commit

        # ── 执行 ──
        try:
            result = subprocess.run(
                cmd,
                cwd=str(work_dir),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30.0,
            )
            output_lines = (result.stdout + result.stderr).splitlines()
            truncated = len(output_lines) > input_data.max_lines
            output = "\n".join(output_lines[: input_data.max_lines])
            # Read actions always succeed at the tool level — git's non-zero
            # exit (e.g. "not a git repository") is captured output, not a
            # tool failure. Write actions propagate the exit code as success.
            is_read = input_data.action in _READ_ACTIONS
            success = True if is_read else result.returncode == 0

            # RevertibleEffect (Cordis): 写操作成功后, 把补偿逆登记进
            # 调用方的 RevertibleContext, 上层 workflow/沙箱统一回滚.
            if success and not is_read and context is not None and context.revertible is not None:
                rv = context.revertible
                if input_data.action == "commit":
                    rv.compensate("git_commit", {"work_dir": str(work_dir), "prev_hash": prev_hash})
                elif input_data.action in ("branch_create", "checkout"):
                    rv.compensate("git_checkout", {"work_dir": str(work_dir), "prev_ref": prev_ref})
                elif input_data.action == "add":
                    rv.compensate("git_add", {"work_dir": str(work_dir)})

            return ToolResult(
                data={
                    "action": input_data.action,
                    "command": " ".join(cmd),
                    "output": output,
                    "truncated": truncated,
                    "returncode": result.returncode,
                    "message": f"Git {input_data.action} {'completed' if result.returncode == 0 else 'failed'}.",
                },
                success=success,
                error=None if success else f"Git {input_data.action} failed (exit {result.returncode}): {result.stderr.strip()[:200]}",
            )
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"Git tool failed: {e}")


def _build_cmd(input_data: GitToolInput) -> list[str] | None:
    """Build the git command for the given action. Returns None for bad input."""
    action = input_data.action
    if action == "status":
        return ["git", "status", "--short"]
    if action == "diff":
        return ["git", "diff"]
    if action == "log":
        return ["git", "log", "--oneline", "-20"]
    if action == "add":
        if input_data.paths:
            return ["git", "add", *input_data.paths]
        return ["git", "add", "-A"]
    if action == "commit":
        if not input_data.message:
            return None  # commit 必须带 message
        return ["git", "commit", "-m", input_data.message]
    if action == "branch_create":
        if not input_data.branch:
            return None
        return ["git", "checkout", "-b", input_data.branch]
    if action == "checkout":
        if not input_data.branch:
            return None
        return ["git", "checkout", input_data.branch]
    return None
