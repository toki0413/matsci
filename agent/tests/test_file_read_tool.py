"""FileReadTool 全分支测试 — Rust `tail_lines` 桥接 + call() 集成路径.

huginn_ext 未编译安装时, `_tail_lines` 的 Rust fast path 走不到; 这里用
sys.modules 注入 fake huginn_ext 覆盖 Rust 路径, 并测回退逻辑. 同时覆盖
call() 的路径穿越保护、文件过大、PDF 提取、n_lines/line_offset、非文件、
token cap、异常回退等分支 (原 test_file_read_tool_ext.py 与
test_file_read_tool_integration_ext.py 归并).
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


async def _call(args, monkeypatch=None):
    tool = FileReadTool()
    return await tool.call(args)


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


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


# ── 基础读取 ──────────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_call_basic_read(tmp_path):
    p = _write(tmp_path / "f.txt", "line1\nline2\nline3\n")
    res = await _call({"file_path": str(p)})
    assert res.success is True
    assert res.data["total_lines"] == 3
    assert "line2" in res.data["content"]


@pytest.mark.anyio
async def test_call_line_offset_and_n_lines(tmp_path):
    p = _write(tmp_path / "f.txt", "l1\nl2\nl3\nl4\nl5\n")
    res = await _call({"file_path": str(p), "line_offset": 2, "n_lines": 2})
    assert res.data["start_line"] == 2
    assert "l2" in res.data["content"]
    assert "l3" in res.data["content"]
    assert "l5" not in res.data["content"]


@pytest.mark.anyio
async def test_call_relative_path(tmp_path, monkeypatch):
    # working_dir 指定 + 相对路径
    p = _write(tmp_path / "rel.txt", "hi\n")
    res = await _call({"file_path": "rel.txt", "working_dir": str(tmp_path)})
    assert res.success is True


# ── 路径穿越保护 ──────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_call_path_traversal_denied(tmp_path, monkeypatch):
    # 绝对路径越界到 working_dir 之外 → 拒绝 (conftest 默认开 unrestricted, 需关)
    monkeypatch.delenv("HUGINN_ALLOW_UNRESTRICTED_READ", raising=False)
    secret = _write(tmp_path / "secret.txt", "secret\n")
    sub = tmp_path / "sub"
    sub.mkdir()
    res = await _call({"file_path": str(secret), "working_dir": str(sub)})
    assert res.success is False
    assert "Access denied" in res.error


@pytest.mark.anyio
async def test_call_unrestricted_read_env(monkeypatch, tmp_path):
    # HUGINN_ALLOW_UNRESTRICTED_READ=1 → 允许越界
    monkeypatch.setenv("HUGINN_ALLOW_UNRESTRICTED_READ", "1")
    outside = _write(tmp_path / "o.txt", "outside\n")
    res = await _call({"file_path": str(outside)})
    assert res.success is True


# ── 文件不存在 / 非文件 / 过大 ────────────────────────────────────────────


@pytest.mark.anyio
async def test_call_file_not_found(tmp_path):
    res = await _call({"file_path": str(tmp_path / "nope.txt")})
    assert res.success is False
    assert "File not found" in res.error


@pytest.mark.anyio
async def test_call_dir_not_file(tmp_path):
    res = await _call({"file_path": str(tmp_path)})
    assert res.success is False
    assert "Not a file" in res.error


@pytest.mark.anyio
async def test_call_file_too_large(tmp_path):
    p = _write(tmp_path / "big.txt", "x" * 5000 + "\n")
    res = await _call({"file_path": str(p), "max_size_bytes": 100})
    assert res.success is False
    assert "File too large" in res.error


# ── token cap ─────────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_call_token_truncation(tmp_path):
    p = _write(tmp_path / "t.txt", "".join(f"row {i}\n" for i in range(50)))
    res = await _call({"file_path": str(p), "max_output_tokens": 5})
    assert res.success is True
    assert "Truncated" in res.data["message"]


@pytest.mark.anyio
async def test_call_env_max_size(monkeypatch, tmp_path):
    monkeypatch.setenv("HUGINN_FILE_READ_MAX_SIZE_BYTES", "10")
    p = _write(tmp_path / "f.txt", "this is a long line\n")
    res = await _call({"file_path": str(p)})
    assert res.success is False
    assert "File too large" in res.error


# ── PDF 提取 ──────────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_call_pdf_success(monkeypatch, tmp_path):
    # fake _extract_text 返回文本
    from huginn.knowledge import store as kb

    def fake_extract(path, content):
        return "method a\nresult b\nconclusion c\n"

    monkeypatch.setattr(kb, "_extract_text", fake_extract)
    p = _write(tmp_path / "paper.pdf", "%PDF fake")
    res = await _call({"file_path": str(p)})
    assert res.success is True
    assert "PDF" in res.data["message"]
    assert "result b" in res.data["content"]


@pytest.mark.anyio
async def test_call_pdf_failure(monkeypatch, tmp_path):
    from huginn.knowledge import store as kb

    def bad_extract(path, content):
        raise RuntimeError("pymupdf missing")

    monkeypatch.setattr(kb, "_extract_text", bad_extract)
    p = _write(tmp_path / "pp.pdf", "%PDF")
    res = await _call({"file_path": str(p)})
    assert res.success is False
    assert "PDF extraction failed" in res.error


@pytest.mark.anyio
async def test_call_pdf_empty_text(monkeypatch, tmp_path):
    from huginn.knowledge import store as kb

    monkeypatch.setattr(kb, "_extract_text", lambda path, content: "")
    p = _write(tmp_path / "e.pdf", "%PDF")
    # 空文本 → pdf_text falsy → 落回普通 read_text 分支
    res = await _call({"file_path": str(p)})
    assert res.success is True


# ── 异常回退 ──────────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_call_read_exception(monkeypatch, tmp_path):
    # 让 stat 抛异常 → 走 except 分支
    from pathlib import Path

    p = _write(tmp_path / "f.txt", "x\n")
    real_exists = Path.exists
    real_is_file = Path.is_file
    real_stat = Path.stat

    # 只对目标文件拦截 stat, 其余路径（含 pytest 内部 pathlib 调用）走真实实现,
    # 避免全局 monkeypatch Path.stat 破坏 pytest 自身（Python 3.12+ 的
    # exists()/is_file() 会带 follow_symlinks 关键字调 stat）.
    def fake_exists(self, *args, **kwargs):
        return True if self == p else real_exists(self, *args, **kwargs)

    def fake_is_file(self, *args, **kwargs):
        return True if self == p else real_is_file(self, *args, **kwargs)

    def selective_stat(self, *args, **kwargs):
        if self == p:
            raise OSError("io error")
        return real_stat(self, *args, **kwargs)

    # exists/is_file 对目标文件返回 True, 让执行推进到 try 块内的 stat().
    monkeypatch.setattr(Path, "exists", fake_exists)
    monkeypatch.setattr(Path, "is_file", fake_is_file)
    monkeypatch.setattr(Path, "stat", selective_stat)
    res = await _call({"file_path": str(p)})
    assert res.success is False
    assert "Failed to read file" in res.error


# ── _apply_token_cap 直接测 ───────────────────────────────────────────────


def test_apply_token_cap_no_truncation():
    tool = FileReadTool()
    lines, truncated = tool._apply_token_cap(["a", "b"], 1, 10000, ".txt")
    assert truncated is False
    assert lines == ["a", "b"]


def test_apply_token_cap_truncates():
    tool = FileReadTool()
    lines, truncated = tool._apply_token_cap(["x" * 5000, "y"], 1, 5, ".txt")
    assert truncated is True
    assert len(lines) < 2