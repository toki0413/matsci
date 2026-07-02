"""GitHub tool — Issue/PR workflow via the ``gh`` CLI.

Wraps ``gh issue``, ``gh pr``, and ``gh checks`` so the agent can read issues,
open PRs, and inspect CI status without leaving the conversation. Degrades
gracefully to a clear error when ``gh`` is not installed or not authenticated.

All write actions (pr_create, pr_comment) are destructive and gated by the
permission system. Read actions (issue_get, issue_list, pr_view, checks_view)
are auto-approved.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.permissions import PermissionChecker, PermissionConfig
from huginn.tools.base import HuginnTool
from huginn.tools.profile import ToolProfile
from huginn.types import PermissionMode, ToolContext, ToolResult


class GithubToolInput(BaseModel):
    action: Literal[
        "issue_get",
        "issue_list",
        "pr_create",
        "pr_view",
        "pr_comment",
        "checks_view",
    ] = Field(...)
    working_dir: str | None = Field(default=None)
    repo: str | None = Field(default=None, description="owner/repo. None = infer from git remote.")
    # issue params
    issue_number: int | None = Field(default=None)
    # pr params
    title: str | None = Field(default=None, description="PR title (pr_create).")
    body: str | None = Field(default=None, description="PR body (pr_create) or comment body (pr_comment).")
    base: str | None = Field(default=None, description="Target branch for PR (pr_create). Defaults to repo default.")
    head: str | None = Field(default=None, description="Source branch for PR (pr_create). Defaults to current.")
    pr_number: int | None = Field(default=None, description="PR number (pr_view/pr_comment).")
    state: Literal["open", "closed", "all"] = Field(default="open", description="Filter for issue_list.")
    limit: int = Field(default=30, ge=1, le=100)


_READ_ACTIONS = {"issue_get", "issue_list", "pr_view", "checks_view"}
_WRITE_ACTIONS = {"pr_create", "pr_comment"}


def _gh_available() -> bool:
    """True if ``gh`` CLI is on PATH."""
    return shutil.which("gh") is not None


def _infer_repo(work_dir: Path) -> str | None:
    """Try to get owner/repo from git remote."""
    try:
        result = subprocess.run(
            ["gh", "repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"],
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            timeout=15.0,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


class GithubTool(HuginnTool):
    """GitHub Issue/PR workflow via ``gh`` CLI."""

    name = "github_tool"
    category = "core"
    description = "Interact with GitHub: read issues, create PRs, view checks. Requires gh CLI."
    input_schema = GithubToolInput
    profile = ToolProfile(cost_tier="light")

    def is_read_only(self, args: GithubToolInput) -> bool:
        return args.action in _READ_ACTIONS

    def is_destructive(self, args: GithubToolInput) -> bool:
        return args.action in _WRITE_ACTIONS

    async def call(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        input_data = GithubToolInput(**args)
        work_dir = (
            Path(input_data.working_dir) if input_data.working_dir else Path.cwd()
        )

        # ── 0. gh 可用性检查 ──
        if not _gh_available():
            return ToolResult(
                data=None,
                success=False,
                error=(
                    "gh CLI not found on PATH. Install from https://cli.github.com/ "
                    "or use git_tool for local-only operations."
                ),
            )

        # ── 1. 推断 repo ──
        repo = input_data.repo or _infer_repo(work_dir)
        if repo is None and input_data.action != "issue_list":
            return ToolResult(
                data=None,
                success=False,
                error="Could not infer owner/repo from git remote. Pass repo explicitly.",
            )

        # ── 2. 构造命令 ──
        cmd = _build_gh_cmd(input_data, repo)
        if cmd is None:
            return ToolResult(
                data=None,
                success=False,
                error=f"Incomplete or unknown github action: {input_data.action}",
            )

        # ── 3. 写动作过权限 ──
        if input_data.action in _WRITE_ACTIONS:
            perm_config = (
                context.config
                if context is not None and isinstance(context.config, PermissionConfig)
                else PermissionConfig()
            )
            checker = PermissionChecker(perm_config)
            perm_result = await checker.check(
                "github_tool",
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
                        "message": f"GitHub {input_data.action} requires approval — no changes made.",
                    },
                    success=True,
                )
            if perm_result.mode == PermissionMode.DENY:
                return ToolResult(
                    data=None,
                    success=False,
                    error=perm_result.reason or f"GitHub {input_data.action} denied.",
                )

        # ── 4. 执行 ──
        try:
            result = subprocess.run(
                cmd,
                cwd=str(work_dir),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60.0,
            )
            success = result.returncode == 0
            # 优先解析 JSON 输出
            data: dict[str, Any] = {"action": input_data.action, "command": " ".join(cmd)}
            if result.stdout.strip().startswith("{") or result.stdout.strip().startswith("["):
                try:
                    data["result"] = json.loads(result.stdout)
                except json.JSONDecodeError:
                    data["output"] = result.stdout.strip()
            else:
                data["output"] = result.stdout.strip()
            data["returncode"] = result.returncode
            data["message"] = f"GitHub {input_data.action} {'completed' if success else 'failed'}."
            return ToolResult(
                data=data,
                success=success,
                error=None if success else f"gh {input_data.action} failed (exit {result.returncode}): {result.stderr.strip()[:300]}",
            )
        except subprocess.TimeoutExpired:
            return ToolResult(data=None, success=False, error=f"gh {input_data.action} timed out.")
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"GitHub tool failed: {e}")


def _build_gh_cmd(input_data: GithubToolInput, repo: str | None) -> list[str] | None:
    """Build the gh command. Returns None for bad input."""
    action = input_data.action
    if action == "issue_get":
        if input_data.issue_number is None or repo is None:
            return None
        return ["gh", "issue", "view", str(input_data.issue_number), "--repo", repo, "--json", "number,title,body,state,labels"]
    if action == "issue_list":
        cmd = ["gh", "issue", "list", "--state", input_data.state, "--limit", str(input_data.limit), "--json", "number,title,state,labels"]
        if repo:
            cmd.extend(["--repo", repo])
        return cmd
    if action == "pr_create":
        if not input_data.title or repo is None:
            return None
        cmd = ["gh", "pr", "create", "--repo", repo, "--title", input_data.title]
        if input_data.body:
            cmd.extend(["--body", input_data.body])
        if input_data.base:
            cmd.extend(["--base", input_data.base])
        if input_data.head:
            cmd.extend(["--head", input_data.head])
        return cmd
    if action == "pr_view":
        if input_data.pr_number is None or repo is None:
            return None
        return ["gh", "pr", "view", str(input_data.pr_number), "--repo", repo, "--json", "number,title,state,url,mergeable,statusCheckRollup"]
    if action == "pr_comment":
        if input_data.pr_number is None or not input_data.body or repo is None:
            return None
        return ["gh", "pr", "comment", str(input_data.pr_number), "--repo", repo, "--body", input_data.body]
    if action == "checks_view":
        if input_data.pr_number is None or repo is None:
            return None
        return ["gh", "pr", "checks", str(input_data.pr_number), "--repo", repo]
    return None
