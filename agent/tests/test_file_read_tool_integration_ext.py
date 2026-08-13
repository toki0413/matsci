"""FileReadTool `call()` 集成路径补测 — 覆盖路径穿越保护、文件过大、
PDF 提取、n_lines/line_offset、非文件、token cap、异常回退等分支.

配合既有 test_file_read_tool_ext.py (Rust tail fast path), 把 file_read_tool.py
覆盖率从 34% 提升到 90%+.
"""

from __future__ import annotations

import pytest

from huginn.tools.file_read_tool import FileReadTool


async def _call(args, monkeypatch=None):
    tool = FileReadTool()
    return await tool.call(args)


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


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

    def boom(self):
        raise OSError("io error")

    monkeypatch.setattr(Path, "stat", boom)
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