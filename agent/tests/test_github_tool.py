"""Tests for GithubTool — GitHub Issue/PR workflow via gh CLI."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

import pytest

from huginn.permissions import PermissionConfig
from huginn.tools.github_tool import GithubTool, _build_gh_cmd
from huginn.types import ToolContext


@pytest.fixture
def tool():
    return GithubTool()


@pytest.fixture
def ctx(tmp_path):
    # auto_approve_all=True 让写动作直接放行, 绕过 ASK 预览分支
    return ToolContext(
        session_id="test",
        workspace=str(tmp_path),
        config=PermissionConfig(auto_approve_all=True),
    )


@pytest.fixture
def ctx_ask(tmp_path):
    # 默认 ASK 模式, 写动作应返回 dry_run
    return ToolContext(
        session_id="test",
        workspace=str(tmp_path),
        config=PermissionConfig(auto_approve_all=False),
    )


# ── _build_gh_cmd: 纯函数命令构造 ──


class TestBuildGhCmd:
    def test_issue_get_requires_number(self):
        from huginn.tools.github_tool import GithubToolInput
        inp = GithubToolInput(action="issue_get", repo="owner/repo")
        assert _build_gh_cmd(inp, "owner/repo") is None  # 缺 issue_number

    def test_issue_get_constructs_cmd(self):
        from huginn.tools.github_tool import GithubToolInput
        inp = GithubToolInput(action="issue_get", issue_number=42, repo="owner/repo")
        cmd = _build_gh_cmd(inp, "owner/repo")
        assert cmd[0] == "gh"
        assert "issue" in cmd
        assert "view" in cmd
        assert "42" in cmd
        assert "owner/repo" in cmd

    def test_issue_list_with_repo(self):
        from huginn.tools.github_tool import GithubToolInput
        inp = GithubToolInput(action="issue_list", repo="owner/repo")
        cmd = _build_gh_cmd(inp, "owner/repo")
        assert "list" in cmd
        assert "owner/repo" in cmd

    def test_pr_create_requires_title(self):
        from huginn.tools.github_tool import GithubToolInput
        inp = GithubToolInput(action="pr_create", repo="owner/repo")
        assert _build_gh_cmd(inp, "owner/repo") is None  # 缺 title

    def test_pr_create_full_cmd(self):
        from huginn.tools.github_tool import GithubToolInput
        inp = GithubToolInput(
            action="pr_create",
            repo="owner/repo",
            title="Fix bug",
            body="desc",
            base="main",
            head="feature",
        )
        cmd = _build_gh_cmd(inp, "owner/repo")
        assert "create" in cmd
        assert "Fix bug" in cmd
        assert "main" in cmd
        assert "feature" in cmd

    def test_pr_view_requires_number(self):
        from huginn.tools.github_tool import GithubToolInput
        inp = GithubToolInput(action="pr_view", repo="owner/repo")
        assert _build_gh_cmd(inp, "owner/repo") is None

    def test_pr_comment_requires_body(self):
        from huginn.tools.github_tool import GithubToolInput
        inp = GithubToolInput(action="pr_comment", pr_number=1, repo="owner/repo")
        assert _build_gh_cmd(inp, "owner/repo") is None  # 缺 body


# ── gh 缺失降级 ──


def test_gh_missing_returns_friendly_error(tool, ctx):
    """gh CLI 不存在时返回明确错误, 不崩溃."""
    with patch("huginn.tools.github_tool._gh_available", return_value=False):
        result = asyncio.run(
            tool.call(
                {"action": "issue_list", "working_dir": ctx.workspace},
                ctx,
            )
        )
    assert result.success is False
    assert "gh CLI not found" in result.error


# ── 读动作: mock subprocess ──


def test_issue_get_parses_json(tool, ctx):
    """issue_get 读动作: mock gh 返回 JSON, 验证解析."""
    gh_output = json.dumps({
        "number": 42,
        "title": "Bug report",
        "body": "Something broken",
        "state": "OPEN",
        "labels": [{"name": "bug"}],
    })

    with patch("huginn.tools.github_tool._gh_available", return_value=True), \
         patch("huginn.tools.github_tool._infer_repo", return_value="owner/repo"), \
         patch("huginn.tools.github_tool.subprocess.run") as mock_run:
        mock_run.return_value = type("R", (), {
            "returncode": 0,
            "stdout": gh_output,
            "stderr": "",
        })()
        result = asyncio.run(
            tool.call(
                {"action": "issue_get", "issue_number": 42, "working_dir": ctx.workspace},
                ctx,
            )
        )
    assert result.success is True
    assert result.data["result"]["number"] == 42
    assert result.data["result"]["title"] == "Bug report"


def test_pr_view_parses_json(tool, ctx):
    """pr_view 读动作: mock gh 返回 PR JSON."""
    gh_output = json.dumps({
        "number": 7,
        "title": "Add feature",
        "state": "OPEN",
        "url": "https://github.com/owner/repo/pull/7",
        "mergeable": True,
    })

    with patch("huginn.tools.github_tool._gh_available", return_value=True), \
         patch("huginn.tools.github_tool._infer_repo", return_value="owner/repo"), \
         patch("huginn.tools.github_tool.subprocess.run") as mock_run:
        mock_run.return_value = type("R", (), {
            "returncode": 0,
            "stdout": gh_output,
            "stderr": "",
        })()
        result = asyncio.run(
            tool.call(
                {"action": "pr_view", "pr_number": 7, "working_dir": ctx.workspace},
                ctx,
            )
        )
    assert result.success is True
    assert result.data["result"]["number"] == 7
    assert result.data["result"]["mergeable"] is True


# ── 写动作: 权限检查 ──


def test_pr_create_ask_mode_returns_dry_run(tool, ctx_ask):
    """pr_create 在 ASK 模式下返回 dry_run, 不执行 gh."""
    with patch("huginn.tools.github_tool._gh_available", return_value=True), \
         patch("huginn.tools.github_tool._infer_repo", return_value="owner/repo"), \
         patch("huginn.tools.github_tool.subprocess.run") as mock_run:
        result = asyncio.run(
            tool.call(
                {
                    "action": "pr_create",
                    "title": "Test PR",
                    "working_dir": ctx_ask.workspace,
                },
                ctx_ask,
            )
        )
    assert result.success is True
    assert result.data["dry_run"] is True
    assert result.data["needs_approval"] is True
    # gh 不应被调用
    mock_run.assert_not_called()


def test_pr_create_auto_mode_executes(tool, ctx):
    """pr_create 在 auto_approve_all=True 下实际执行 gh."""
    gh_output = json.dumps({"number": 99, "url": "https://github.com/owner/repo/pull/99"})

    with patch("huginn.tools.github_tool._gh_available", return_value=True), \
         patch("huginn.tools.github_tool._infer_repo", return_value="owner/repo"), \
         patch("huginn.tools.github_tool.subprocess.run") as mock_run:
        mock_run.return_value = type("R", (), {
            "returncode": 0,
            "stdout": gh_output,
            "stderr": "",
        })()
        result = asyncio.run(
            tool.call(
                {
                    "action": "pr_create",
                    "title": "Test PR",
                    "body": "desc",
                    "working_dir": ctx.workspace,
                },
                ctx,
            )
        )
    assert result.success is True
    # auto 模式下不返回 dry_run, 直接执行
    assert result.data.get("dry_run", False) is False
    # 确认 gh 被调用了
    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert "create" in cmd


def test_pr_comment_ask_mode_returns_dry_run(tool, ctx_ask):
    """pr_comment 在 ASK 模式下返回 dry_run."""
    with patch("huginn.tools.github_tool._gh_available", return_value=True), \
         patch("huginn.tools.github_tool._infer_repo", return_value="owner/repo"), \
         patch("huginn.tools.github_tool.subprocess.run") as mock_run:
        result = asyncio.run(
            tool.call(
                {
                    "action": "pr_comment",
                    "pr_number": 5,
                    "body": "Looks good",
                    "working_dir": ctx_ask.workspace,
                },
                ctx_ask,
            )
        )
    assert result.data["dry_run"] is True
    mock_run.assert_not_called()


# ── repo 推断失败 ──


def test_repo_inference_failure_returns_error(tool, ctx):
    """repo 推断失败且未显式传入 repo 时返回友好错误."""
    with patch("huginn.tools.github_tool._gh_available", return_value=True), \
         patch("huginn.tools.github_tool._infer_repo", return_value=None):
        result = asyncio.run(
            tool.call(
                {"action": "issue_get", "issue_number": 1, "working_dir": ctx.workspace},
                ctx,
            )
        )
    assert result.success is False
    assert "Could not infer" in result.error


# ── gh 命令失败 ──


def test_gh_command_failure_returns_error(tool, ctx):
    """gh 退出码非 0 时返回失败."""
    with patch("huginn.tools.github_tool._gh_available", return_value=True), \
         patch("huginn.tools.github_tool._infer_repo", return_value="owner/repo"), \
         patch("huginn.tools.github_tool.subprocess.run") as mock_run:
        mock_run.return_value = type("R", (), {
            "returncode": 1,
            "stdout": "",
            "stderr": "ERROR: not authenticated",
        })()
        result = asyncio.run(
            tool.call(
                {"action": "issue_list", "working_dir": ctx.workspace},
                ctx,
            )
        )
    assert result.success is False
    assert "failed" in result.error.lower()
