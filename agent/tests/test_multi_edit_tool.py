"""Tests for MultiEditTool — atomic multi-file string replacement."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from huginn.permissions import PermissionConfig
from huginn.tools.multi_edit_tool import MultiEditTool
from huginn.types import ToolContext


@pytest.fixture
def tool():
    return MultiEditTool()


@pytest.fixture
def ctx(tmp_path):
    # auto_approve_all=True 让写动作直接放行, 绕过 ASK 预览分支
    return ToolContext(
        session_id="test",
        workspace=str(tmp_path),
        config=PermissionConfig(auto_approve_all=True),
    )


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_atomic_batch_applies_to_two_files(tool, tmp_path, ctx):
    """两个文件各一处替换, 全部通过, 都落盘."""
    f1 = tmp_path / "a.py"
    f2 = tmp_path / "b.py"
    _write(f1, "def foo():\n    return 1\n")
    _write(f2, "def bar():\n    return 2\n")

    result = asyncio.run(
        tool.call(
            {
                "edits": [
                    {"file_path": str(f1), "old_string": "return 1", "new_string": "return 10"},
                    {"file_path": str(f2), "old_string": "return 2", "new_string": "return 20"},
                ],
                "working_dir": str(tmp_path),
            },
            ctx,
        )
    )
    assert result.success is True
    assert result.data["files_changed"] == 2
    assert "return 10" in f1.read_text()
    assert "return 20" in f2.read_text()
    for entry in result.data["applied"]:
        assert "snapshot_hash" in entry
        assert "diff" in entry


def test_atomic_batch_rolls_back_when_one_edit_fails(tool, tmp_path, ctx):
    """其中一个 old_string 出现两次 → 整批不写入, 已校验通过的文件保持原样."""
    f1 = tmp_path / "a.py"
    f2 = tmp_path / "b.py"
    _write(f1, "x = 1\nx = 1\n")
    _write(f2, "y = 2\n")

    result = asyncio.run(
        tool.call(
            {
                "edits": [
                    {"file_path": str(f1), "old_string": "x = 1", "new_string": "x = 99"},
                    {"file_path": str(f2), "old_string": "y = 2", "new_string": "y = 99"},
                ],
                "working_dir": str(tmp_path),
            },
            ctx,
        )
    )
    assert result.success is False
    assert "occurs 2 times" in result.error
    assert f2.read_text() == "y = 2\n"


def test_path_traversal_refused(tool, tmp_path, ctx):
    """越界路径直接拒."""
    outside = tmp_path.parent / "outside_target.txt"
    _write(outside, "secret\n")
    try:
        result = asyncio.run(
            tool.call(
                {
                    "edits": [
                        {"file_path": str(outside), "old_string": "secret", "new_string": "x"},
                    ],
                    "working_dir": str(tmp_path),
                },
                ctx,
            )
        )
        assert result.success is False
        assert "outside working directory" in result.error
        assert outside.read_text() == "secret\n"
    finally:
        if outside.exists():
            outside.unlink()


def test_dry_run_returns_diffs_without_writing(tool, tmp_path, ctx):
    """dry_run 只返回 diff, 文件不动."""
    f = tmp_path / "a.py"
    _write(f, "def foo():\n    return 1\n")
    result = asyncio.run(
        tool.call(
            {
                "edits": [
                    {"file_path": str(f), "old_string": "return 1", "new_string": "return 42"},
                ],
                "working_dir": str(tmp_path),
                "dry_run": True,
            },
            ctx,
        )
    )
    assert result.success is True
    assert result.data["dry_run"] is True
    assert f.read_text() == "def foo():\n    return 1\n"
    assert "return 42" in result.data["edits"][0]["diff"]


def test_file_not_found_aborts(tool, tmp_path, ctx):
    """文件不存在 → 整批中止."""
    result = asyncio.run(
        tool.call(
            {
                "edits": [
                    {"file_path": "nope.py", "old_string": "a", "new_string": "b"},
                ],
                "working_dir": str(tmp_path),
            },
            ctx,
        )
    )
    assert result.success is False
    assert "file not found" in result.error.lower()


def test_two_edits_same_file_both_applied(tool, tmp_path, ctx):
    """同一文件两处替换, 各 old_string 唯一 → 都落盘."""
    f = tmp_path / "a.py"
    _write(f, "def foo():\n    return 1\n\ndef bar():\n    return 2\n")
    result = asyncio.run(
        tool.call(
            {
                "edits": [
                    {"file_path": str(f), "old_string": "return 1", "new_string": "return 10"},
                    {"file_path": str(f), "old_string": "return 2", "new_string": "return 20"},
                ],
                "working_dir": str(tmp_path),
            },
            ctx,
        )
    )
    assert result.success is True
    content = f.read_text()
    assert "return 10" in content
    assert "return 20" in content


def test_large_file_refused(tool, tmp_path, ctx):
    """超 50MB 的文件拒绝编辑."""
    f = tmp_path / "big.txt"
    chunk = "x" * 1024
    with f.open("w", encoding="utf-8") as fh:
        for _ in range(50 * 1024 + 10):
            fh.write(chunk + "\n")
    try:
        result = asyncio.run(
            tool.call(
                {
                    "edits": [
                        {"file_path": str(f), "old_string": "xxx", "new_string": "yyy"},
                    ],
                    "working_dir": str(tmp_path),
                },
                ctx,
            )
        )
        assert result.success is False
        assert "too large" in result.error.lower()
    finally:
        if f.exists():
            f.unlink()


def test_preview_action_returns_diffs(tool, tmp_path, ctx):
    """action=preview 等价 dry_run."""
    f = tmp_path / "a.py"
    _write(f, "value = 1\n")
    result = asyncio.run(
        tool.call(
            {
                "action": "preview",
                "edits": [
                    {"file_path": str(f), "old_string": "value = 1", "new_string": "value = 2"},
                ],
                "working_dir": str(tmp_path),
            },
            ctx,
        )
    )
    assert result.success is True
    assert result.data["dry_run"] is True
    assert f.read_text() == "value = 1\n"


def test_empty_edits_rejected_by_schema(tool, tmp_path, ctx):
    """空 edits 列表 schema 层拒绝."""
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        asyncio.run(
            tool.call(
                {"edits": [], "working_dir": str(tmp_path)},
                ctx,
            )
        )
