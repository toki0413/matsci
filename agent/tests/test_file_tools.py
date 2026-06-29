"""Tests for core 类 4 个文件/git 工具 — 行为测试, 不依赖 LLM.

覆盖: FileEdit / FileRead / FileWrite / Git
风格参考 test_design_tools.py / test_meta_tools.py.
文件系统测试用 tmp_path fixture 自动清理.
GitTool 用 tmp_path + subprocess 造真仓, shutil.which("git") skip guard.
FileEditTool 写入分支需要 PermissionConfig(auto_approve_all=True) 绕过 ASK.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from huginn.permissions import PermissionConfig
from huginn.tools.file_edit_tool import FileEditTool
from huginn.tools.file_read_tool import FileReadTool
from huginn.tools.file_write_tool import FileWriteTool
from huginn.tools.git_tool import GitTool


# git 不在就跳过 git 那一组测试
_HAS_GIT = shutil.which("git") is not None
_skip_no_git = pytest.mark.skipif(not _HAS_GIT, reason="git binary not on PATH")


def _git_init_repo(repo_dir: Path) -> None:
    """在 tmp_path 里造一个真 git 仓, 配好 user + 一个初始 commit."""
    subprocess.run(
        ["git", "init"], cwd=str(repo_dir), capture_output=True, check=True
    )
    subprocess.run(
        ["git", "config", "user.email", "test@test.local"],
        cwd=str(repo_dir),
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=str(repo_dir),
        capture_output=True,
        check=True,
    )


def _git_commit(repo_dir: Path, msg: str) -> None:
    subprocess.run(
        ["git", "add", "-A"], cwd=str(repo_dir), capture_output=True, check=True
    )
    subprocess.run(
        ["git", "commit", "-m", msg],
        cwd=str(repo_dir),
        capture_output=True,
        check=True,
    )


# ════════════════════════════════════════════════════════════════════
# FileEditTool
# ════════════════════════════════════════════════════════════════════


async def test_edit_path_traversal_rejected(tmp_path):
    """绝对路径在 work_dir 之外 → success=False 'outside working directory'."""
    # 用绝对路径测越界, pathlib 的 .. 不会被 relative_to 识别为越界
    outside = tmp_path.parent / "outside_edit_target.txt"
    tool = FileEditTool()
    result = await tool.call(
        {
            "file_path": str(outside),
            "old_string": "a",
            "new_string": "b",
            "working_dir": str(tmp_path),
        },
        context=None,
    )
    assert result.success is False
    assert "outside working directory" in result.error


async def test_edit_file_not_exist_returns_error(tmp_path):
    """文件不存在 → success=False 'File not found'."""
    tool = FileEditTool()
    result = await tool.call(
        {
            "file_path": "ghost.txt",
            "old_string": "a",
            "new_string": "b",
            "working_dir": str(tmp_path),
        },
        context=None,
    )
    assert result.success is False
    assert "File not found" in result.error


async def test_edit_old_string_not_unique_returns_error(tmp_path):
    """old_string 在文件里出现 2 次 → success=False."""
    (tmp_path / "dup.txt").write_text("x\nx\n", encoding="utf-8")
    tool = FileEditTool()
    result = await tool.call(
        {
            "file_path": "dup.txt",
            "old_string": "x",
            "new_string": "y",
            "working_dir": str(tmp_path),
        },
        context=None,
    )
    assert result.success is False
    assert "occurs 2 times" in result.error


async def test_edit_dry_run_returns_diff_no_write(tmp_path):
    """dry_run=True 返回 diff, 文件不变."""
    f = tmp_path / "src.txt"
    original = "line1\nold\nline3\n"
    f.write_text(original, encoding="utf-8")
    tool = FileEditTool()
    result = await tool.call(
        {
            "file_path": "src.txt",
            "old_string": "old",
            "new_string": "new",
            "working_dir": str(tmp_path),
            "dry_run": True,
        },
        context=None,
    )
    assert result.success is True
    assert result.data["dry_run"] is True
    assert "-old" in result.data["diff"]
    assert "+new" in result.data["diff"]
    # 文件没被改
    assert f.read_text(encoding="utf-8") == original


async def test_edit_normal_writes_file(tmp_path):
    """auto_approve 放行后, edit 真写入文件, 返回 snapshot_hash."""
    f = tmp_path / "src.txt"
    f.write_text("hello world\n", encoding="utf-8")
    tool = FileEditTool()
    ctx = SimpleNamespace(config=PermissionConfig(auto_approve_all=True))
    result = await tool.call(
        {
            "file_path": "src.txt",
            "old_string": "hello",
            "new_string": "goodbye",
            "working_dir": str(tmp_path),
        },
        context=ctx,
    )
    assert result.success is True
    assert result.data["replacements"] == 1
    assert "snapshot_hash" in result.data
    assert f.read_text(encoding="utf-8") == "goodbye world\n"


# ════════════════════════════════════════════════════════════════════
# FileReadTool
# ════════════════════════════════════════════════════════════════════


async def test_read_file_not_exist_returns_error(tmp_path):
    """文件不存在 → success=False 'File not found'."""
    tool = FileReadTool()
    result = await tool.call(
        {"file_path": "ghost.txt", "working_dir": str(tmp_path)},
        context=None,
    )
    assert result.success is False
    assert "File not found" in result.error


async def test_read_tail_mode_returns_last_n_lines(tmp_path):
    """tail_lines=3 返回最后 3 行 + start_line 正确."""
    f = tmp_path / "log.txt"
    f.write_text("\n".join(f"line{i}" for i in range(1, 11)) + "\n", encoding="utf-8")
    tool = FileReadTool()
    result = await tool.call(
        {
            "file_path": "log.txt",
            "tail_lines": 3,
            "working_dir": str(tmp_path),
        },
        context=None,
    )
    assert result.success is True
    # 最后 3 行是 line8/line9/line10
    assert "line10" in result.data["content"]
    assert "line8" in result.data["content"]
    assert "line7" not in result.data["content"]


async def test_read_offset_mode_returns_from_line(tmp_path):
    """line_offset=5, n_lines=2 → 返回第 5-6 行."""
    f = tmp_path / "src.txt"
    f.write_text("\n".join(f"line{i}" for i in range(1, 11)) + "\n", encoding="utf-8")
    tool = FileReadTool()
    result = await tool.call(
        {
            "file_path": "src.txt",
            "line_offset": 5,
            "n_lines": 2,
            "working_dir": str(tmp_path),
        },
        context=None,
    )
    assert result.success is True
    assert result.data["start_line"] == 5
    assert "line5" in result.data["content"]
    assert "line6" in result.data["content"]
    assert "line4" not in result.data["content"]
    assert "line7" not in result.data["content"]


async def test_read_max_size_returns_error(tmp_path):
    """文件 > max_size_bytes → success=False 'too large'."""
    f = tmp_path / "big.txt"
    f.write_text("x" * 1000, encoding="utf-8")
    tool = FileReadTool()
    result = await tool.call(
        {
            "file_path": "big.txt",
            "max_size_bytes": 100,
            "working_dir": str(tmp_path),
        },
        context=None,
    )
    assert result.success is False
    assert "too large" in result.error


# ════════════════════════════════════════════════════════════════════
# FileWriteTool
# ════════════════════════════════════════════════════════════════════


async def test_write_path_traversal_rejected(tmp_path):
    """绝对路径在 work_dir 之外 → success=False 'outside working directory'."""
    # 用绝对路径测越界, pathlib 的 .. 不会被 relative_to 识别为越界
    outside = tmp_path.parent / "outside_write_target.txt"
    tool = FileWriteTool()
    result = await tool.call(
        {
            "file_path": str(outside),
            "content": "x",
            "working_dir": str(tmp_path),
        },
        context=None,
    )
    assert result.success is False
    assert "outside working directory" in result.error


async def test_write_dry_run_returns_diff_no_write(tmp_path):
    """dry_run=True 返回 diff, 文件不写."""
    tool = FileWriteTool()
    result = await tool.call(
        {
            "file_path": "new.txt",
            "content": "hello\n",
            "working_dir": str(tmp_path),
            "dry_run": True,
        },
        context=None,
    )
    assert result.success is True
    assert result.data["dry_run"] is True
    assert result.data["existed"] is False
    assert "+hello" in result.data["diff"]
    # 文件没被创建
    assert not (tmp_path / "new.txt").exists()


async def test_write_new_file(tmp_path):
    """写新文件 → success=True, bytes_written 正确."""
    tool = FileWriteTool()
    result = await tool.call(
        {
            "file_path": "fresh.txt",
            "content": "content here\n",
            "working_dir": str(tmp_path),
        },
        context=None,
    )
    assert result.success is True
    assert result.data["existed"] is False
    assert result.data["bytes_written"] == len("content here\n".encode("utf-8"))
    assert (tmp_path / "fresh.txt").read_text(encoding="utf-8") == "content here\n"


async def test_overwrite_existing_returns_snapshot_hash(tmp_path):
    """覆盖已有文件 → 返回 snapshot_hash + existed=True."""
    f = tmp_path / "old.txt"
    f.write_text("old content\n", encoding="utf-8")
    tool = FileWriteTool()
    result = await tool.call(
        {
            "file_path": "old.txt",
            "content": "new content\n",
            "working_dir": str(tmp_path),
        },
        context=None,
    )
    assert result.success is True
    assert result.data["existed"] is True
    assert "snapshot_hash" in result.data
    assert f.read_text(encoding="utf-8") == "new content\n"


# ════════════════════════════════════════════════════════════════════
# GitTool
# ════════════════════════════════════════════════════════════════════


@_skip_no_git
async def test_git_status_on_real_repo(tmp_path):
    """真 git 仓 + 一次 commit → status 返回 success=True."""
    _git_init_repo(tmp_path)
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    _git_commit(tmp_path, "init")
    tool = GitTool()
    result = await tool.call(
        {"action": "status", "working_dir": str(tmp_path)},
        context=None,
    )
    assert result.success is True
    assert result.data["action"] == "status"


@_skip_no_git
async def test_git_diff_returns_changes(tmp_path):
    """修改 tracked 文件后 git diff 返回变更内容."""
    _git_init_repo(tmp_path)
    (tmp_path / "a.txt").write_text("original\n", encoding="utf-8")
    _git_commit(tmp_path, "init")
    # 修改 tracked 文件, 不 commit
    (tmp_path / "a.txt").write_text("modified\n", encoding="utf-8")
    tool = GitTool()
    result = await tool.call(
        {"action": "diff", "working_dir": str(tmp_path)},
        context=None,
    )
    assert result.success is True
    assert "-original" in result.data["output"]
    assert "+modified" in result.data["output"]


@_skip_no_git
async def test_git_log_truncation(tmp_path):
    """3 个 commit + max_lines=1 → truncated=True, output 只有 1 行."""
    _git_init_repo(tmp_path)
    for i in range(3):
        (tmp_path / f"f{i}.txt").write_text(str(i), encoding="utf-8")
        _git_commit(tmp_path, f"commit {i}")
    tool = GitTool()
    result = await tool.call(
        {"action": "log", "working_dir": str(tmp_path), "max_lines": 1},
        context=None,
    )
    assert result.success is True
    assert result.data["truncated"] is True
    # output 截到 1 行
    assert len(result.data["output"].splitlines()) <= 1


@_skip_no_git
async def test_git_status_on_non_git_dir_returns_success(tmp_path):
    """非 git 目录跑 status → git 报错被捕获, 工具仍 success=True + stderr 在 output."""
    tool = GitTool()
    result = await tool.call(
        {"action": "status", "working_dir": str(tmp_path)},
        context=None,
    )
    assert result.success is True  # 工具调用本身没炸
    # git 把错误写到 stderr, 工具拼到 output 里
    assert "not a git repository" in result.data["output"].lower()
