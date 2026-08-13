"""FileReadTool 的 Rust `tail_lines` 桥接测试.

huginn_ext 未编译安装时, `_tail_lines` 的 Rust fast path 走不到; 这里用
sys.modules 注入 fake huginn_ext 覆盖 Rust 路径, 并测回退逻辑.
"""

from __future__ import annotations

import sys
import types

import pytest

from huginn.tools.file_read_tool import FileReadTool


def _install_fake_huginn_ext(monkeypatch, tail_lines=None, raise_import=False):
    """把 fake huginn_ext 装进 sys.modules, 让 `_tail_lines` 的 import 命中.

    raise_import=True 时连 import 都失败, 模拟扩展未安装.
    """
    if raise_import:
        monkeypatch.setitem(sys.modules, "huginn_ext", None)
        return
    ext = types.ModuleType("huginn_ext")
    if tail_lines is not None:
        ext.tail_lines = tail_lines
    monkeypatch.setitem(sys.modules, "huginn_ext", ext)


def _write_lines(path, n=10):
    path.write_text("".join(f"line {i + 1}\n" for i in range(n)), encoding="utf-8")
    return n


# ── Rust fast path ─────────────────────────────────────────────────────────

def test_rust_tail_lines_uses_rust_result(monkeypatch, tmp_path):
    p = tmp_path / "f.txt"
    _write_lines(p, 10)

    def rust_tail(path, n):
        return ["line 9", "line 10"]

    _install_fake_huginn_ext(monkeypatch, tail_lines=rust_tail)
    tool = FileReadTool()
    lines, start = tool._tail_lines(p, 2)
    assert lines == ["line 9", "line 10"]
    # start = max(1, len(lines) - n + 1) = max(1, 2-2+1) = 1
    assert start == 1


def test_rust_tail_lines_start_greater_than_one(monkeypatch, tmp_path):
    p = tmp_path / "f.txt"
    _write_lines(p, 10)

    def rust_tail(path, n):
        return [f"line {i}" for i in range(4, 8)]  # 4 lines

    _install_fake_huginn_ext(monkeypatch, tail_lines=rust_tail)
    tool = FileReadTool()
    lines, start = tool._tail_lines(p, 3)
    assert lines == ["line 4", "line 5", "line 6", "line 7"]
    # start = max(1, 4 - 3 + 1) = 2
    assert start == 2


def test_rust_tail_lines_import_missing_falls_back(monkeypatch, tmp_path):
    p = tmp_path / "f.txt"
    _write_lines(p, 10)
    _install_fake_huginn_ext(monkeypatch, raise_import=True)
    tool = FileReadTool()
    lines, start = tool._tail_lines(p, 3)
    assert lines == ["line 8", "line 9", "line 10"]
    assert start == 8


def test_rust_tail_lines_raises_falls_back(monkeypatch, tmp_path):
    p = tmp_path / "f.txt"
    _write_lines(p, 10)

    def rust_tail(path, n):
        raise RuntimeError("rust crash")

    _install_fake_huginn_ext(monkeypatch, tail_lines=rust_tail)
    tool = FileReadTool()
    lines, start = tool._tail_lines(p, 2)
    assert lines == ["line 9", "line 10"]
    assert start == 9


def test_rust_tail_lines_file_missing_falls_back_error(monkeypatch, tmp_path):
    p = tmp_path / "nodir" / "f.txt"
    _install_fake_huginn_ext(monkeypatch, raise_import=True)
    tool = FileReadTool()
    with pytest.raises(FileNotFoundError):
        tool._tail_lines(p, 2)


# ── 端到端 call() 走 Rust tail ────────────────────────────────────────────

async def test_call_tail_via_rust_fast_path(monkeypatch, tmp_path):
    p = tmp_path / "f.txt"
    _write_lines(p, 10)

    def rust_tail(path, n):
        return ["line 9", "line 10"]

    _install_fake_huginn_ext(monkeypatch, tail_lines=rust_tail)
    tool = FileReadTool()
    res = await tool.call({"file_path": str(p), "tail_lines": 2})
    assert res.success is True
    assert res.data["start_line"] == 1
    assert "line 9" in res.data["content"]
    assert "line 10" in res.data["content"]


async def test_call_tail_falls_back_when_rust_missing(monkeypatch, tmp_path):
    p = tmp_path / "f.txt"
    _write_lines(p, 10)
    _install_fake_huginn_ext(monkeypatch, raise_import=True)
    tool = FileReadTool()
    res = await tool.call({"file_path": str(p), "tail_lines": 3})
    assert res.success is True
    assert res.data["start_line"] == 8
    assert "line 8" in res.data["content"]
    assert "line 10" in res.data["content"]