"""大文件读取稳定性测试.

覆盖:
(a) file_read_tool 的分块窗口懒读按行读取超大文本内容正确;
(b) 窗口模式 / 默认路径下尾部 token 截断仍生效;
(c) routes/fs.py::fs_read 对超大文件返回受保护错误而非整读;
(d) 向后兼容: 不传窗口参数时默认整读路径不回归.

超大文本构造为 >256KB (超过 file_read_tool 的默认整读上限), 验证整读会被
保护拦下, 而窗口懒读能正常按行读出内容. 窗口边界按整行对齐, 便于逐行比对.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

from huginn.tools.file_read_tool import DEFAULT_MAX_SIZE_BYTES, FileReadTool


def _write_big_file(path, n_lines=4000, line="sample content "):
    """构造足够大、行号可校验的文本文件 (每行带 6 位序号)."""
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n_lines):
            f.write(f"{line}{i:06d}\n")
    return path


async def _call(args):
    return await FileReadTool().call(args)


def _bare(content_lines):
    """去掉 numbered 输出的行号前缀 (前 6 字符 `{:4d}  `), 还原正文行."""
    return [cl[6:] for cl in content_lines]


def _load_fs():
    """单独加载 huginn/routes/fs.py, 避开 huginn.routes 包 __init__ 的整树导入.

    huginn.routes.__init__ 会 import 全部路由(进而拉 scipy / python-multipart
    等未装依赖); fs.py 本身只依赖 fastapi/标准库, 从文件单独加载即可测其实现.
    """
    fs_path = Path(__file__).resolve().parent.parent / "huginn" / "routes" / "fs.py"
    spec = importlib.util.spec_from_file_location("huginn_routes_fs", fs_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ── (a) 分块窗口懒读: 超大文本按行读取内容正确 ────────────────────────────


@pytest.mark.anyio
async def test_window_read_walks_huge_file_by_lines(tmp_path):
    """超大文件 (>256KB 整读上限) 用 window_offset/window_size 分窗口读, 内容逐行正确."""
    n_lines = 4000
    line_body = "x" * 81  # 81 + 6 位序号 = 87 字符, + \n = 88 字节/行.
    line_bytes = 88
    per_window = 50
    window_size = line_bytes * per_window  # 每窗口恰好 50 个整行, 边界对齐.
    p = _write_big_file(tmp_path / "big.log", n_lines=n_lines, line=line_body)

    # 整读应被默认 256KB 上限保护拦下, 证明该文件确实超大.
    guard = await _call({"file_path": str(p)})
    assert guard.success is False
    assert "File too large" in guard.error

    # 分窗口遍历: 每窗口只载入 window_size 字节.
    accumulated = []  # 已累计的正文行
    total_size = p.stat().st_size
    offset = 0
    while offset < total_size:
        res = await _call(
            {
                "file_path": str(p),
                "window_offset": offset,
                "window_size": window_size,
            }
        )
        assert res.success is True, res.error
        assert res.data["total_lines"] == n_lines
        assert res.data["window_offset"] == offset
        assert res.data["window_size"] == window_size
        content_lines = res.data["content"].splitlines()
        assert content_lines, "空窗口不应出现"
        # 首行行号 = 已累计行数 + 1 (窗口边界整行对齐, 可直接用长度推算).
        assert res.data["start_line"] == len(accumulated) + 1
        accumulated.extend(_bare(content_lines))
        offset += window_size

    assert accumulated[0] == line_body + "000000"
    expected = [f"{line_body}{i:06d}" for i in range(n_lines)]
    assert accumulated == expected


@pytest.mark.anyio
async def test_window_read_middle_window_content_exact(tmp_path):
    """窗口懒读一个指定字节段, 内容与该段原文逐字节一致."""
    p = _write_big_file(tmp_path / "mid.log", n_lines=100, line="abc ")
    size = p.stat().st_size
    offset = 30
    read_len = 40
    res = await _call(
        {"file_path": str(p), "window_offset": offset, "window_size": read_len}
    )
    assert res.success is True
    raw = p.read_bytes()[offset : offset + read_len].decode("utf-8", "replace")
    expected_lines = raw.splitlines()
    got_lines = _bare(res.data["content"].splitlines())
    assert got_lines == expected_lines
    # 该窗口落在文件中间, 起始行号 > 1.
    assert res.data["start_line"] >= 2
    assert res.data["total_lines"] == 100


# ── (b) 尾部 token 截断仍生效 ─────────────────────────────────────────────


@pytest.mark.anyio
async def test_window_read_token_cap_still_truncates(tmp_path):
    """窗口模式返回大量内容时, 尾部 token 截断仍生效."""
    p = _write_big_file(tmp_path / "trunc.log", n_lines=500, line="fill-")
    res = await _call(
        {
            "file_path": str(p),
            "window_offset": 0,
            "window_size": DEFAULT_MAX_SIZE_BYTES,
            "max_output_tokens": 20,
        }
    )
    assert res.success is True
    assert "Truncated" in res.data["message"]
    assert res.data["start_line"] == 1


@pytest.mark.anyio
async def test_default_path_token_cap_still_truncates(tmp_path):
    """默认整读路径 (无窗口参数) 的尾部 token 截断保持原行为."""
    p = tmp_path / "t.txt"
    p.write_text("".join(f"row {i}\n" for i in range(50)), encoding="utf-8")
    res = await _call({"file_path": str(p), "max_output_tokens": 5})
    assert res.success is True
    assert "Truncated" in res.data["message"]


# ── (c) routes fs_read 对超大文件返回受保护错误 ───────────────────────────


def test_fs_read_protects_large_file(tmp_path, monkeypatch):
    """fs_read 超过阈值应返回保护错误, 而非整读返回内容."""
    monkeypatch.setenv("HUGINN_FS_READ_MAX_SIZE_BYTES", "100")
    p = tmp_path / "big.txt"
    p.write_text("x" * 1000 + "\n", encoding="utf-8")

    fs = _load_fs()
    with pytest.raises(HTTPException) as ei:
        asyncio.run(fs.fs_read(str(p)))
    assert ei.value.status_code == 400
    assert "文件过大" in str(ei.value.detail)
    assert "window_offset" in str(ei.value.detail)


def test_fs_read_small_file_still_works(tmp_path):
    """fs_read 对正常小文件仍正常返回内容 (保护不误伤)."""
    fs = _load_fs()
    p = tmp_path / "ok.txt"
    p.write_text("hello\nworld\n", encoding="utf-8")
    out = asyncio.run(fs.fs_read(str(p)))
    assert out["content"] == "hello\nworld\n"


# ── (d) 向后兼容: 默认路径不回归 ─────────────────────────────────────────


@pytest.mark.anyio
async def test_backward_compat_default_read(tmp_path):
    """无窗口参数时, 默认整读路径保持既有语义 (起始行号/行数与内容)."""
    p = tmp_path / "f.txt"
    p.write_text("L1\nL2\nL3\nL4\n", encoding="utf-8")
    res = await _call({"file_path": str(p)})
    assert res.success is True
    assert res.data["total_lines"] == 4
    assert res.data["start_line"] == 1
    assert "L2" in res.data["content"]

    # line_offset/n_lines 分页语义不变.
    res = await _call({"file_path": str(p), "line_offset": 2, "n_lines": 2})
    assert res.data["start_line"] == 2
    assert "L2" in res.data["content"] and "L4" not in res.data["content"]
