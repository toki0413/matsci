"""P0-2 Hashline 锚定编辑单元测试.

覆盖 file_edit_tool / multi_edit_tool 的 expected_hash + hash_policy:
- strict: 不匹配拒绝, 文件不变
- warn: 不匹配警告但继续
- off / expected_hash=None: 行为与旧版一致
- multi 原子性: 同批一个文件 hash 不匹配整批拒绝
- 闭环: read → edit(带 hash) → 再 edit(用返回 hash) 第二次成功
"""

from __future__ import annotations

import pytest

from huginn.core_types import ToolContext
from huginn.permissions import PermissionConfig
from huginn.tools.file_edit_tool import FileEditTool, _content_hash
from huginn.tools.multi_edit_tool import MultiEditTool


def _ctx(workspace: str = ".") -> ToolContext:
    return ToolContext(
        session_id="t-hashline",
        workspace=workspace,
        config=PermissionConfig(auto_approve_all=True),
    )


# ── file_edit_tool ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_edit_success_returns_hash(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("hello world\n", encoding="utf-8")
    tool = FileEditTool()
    res = await tool.call(
        {"file_path": str(p), "old_string": "world", "new_string": "huginn", "working_dir": str(tmp_path)},
        _ctx(),
    )
    assert res.success is True
    assert res.data["snapshot_hash"] == _content_hash("hello world\n")
    assert p.read_text(encoding="utf-8") == "hello huginn\n"


@pytest.mark.asyncio
async def test_edit_strict_mismatch_refuses(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("hello world\n", encoding="utf-8")
    tool = FileEditTool()
    # 期望一个错误的 hash → strict 拒绝, 文件不变
    res = await tool.call({
        "file_path": str(p),
        "old_string": "world",
        "new_string": "huginn",
        "expected_hash": "deadbeef00000000",
        "hash_policy": "strict",
        "working_dir": str(tmp_path),
    }, _ctx())
    assert res.success is False
    assert "changed since snapshot" in res.error
    assert p.read_text(encoding="utf-8") == "hello world\n"


@pytest.mark.asyncio
async def test_edit_strict_match_allows(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("hello world\n", encoding="utf-8")
    tool = FileEditTool()
    h = _content_hash("hello world\n")
    res = await tool.call({
        "file_path": str(p),
        "old_string": "world",
        "new_string": "huginn",
        "expected_hash": h,
        "hash_policy": "strict",
        "working_dir": str(tmp_path),
    }, _ctx())
    assert res.success is True
    assert p.read_text(encoding="utf-8") == "hello huginn\n"


@pytest.mark.asyncio
async def test_edit_warn_continues_on_mismatch(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("hello world\n", encoding="utf-8")
    tool = FileEditTool()
    res = await tool.call({
        "file_path": str(p),
        "old_string": "world",
        "new_string": "huginn",
        "expected_hash": "deadbeef00000000",
        "hash_policy": "warn",
        "working_dir": str(tmp_path),
    }, _ctx())
    assert res.success is True  # warn 继续写入
    assert p.read_text(encoding="utf-8") == "hello huginn\n"


@pytest.mark.asyncio
async def test_edit_off_skips_check(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("hello world\n", encoding="utf-8")
    tool = FileEditTool()
    res = await tool.call({
        "file_path": str(p),
        "old_string": "world",
        "new_string": "huginn",
        "expected_hash": "deadbeef00000000",
        "hash_policy": "off",
        "working_dir": str(tmp_path),
    }, _ctx())
    assert res.success is True
    assert p.read_text(encoding="utf-8") == "hello huginn\n"


@pytest.mark.asyncio
async def test_edit_no_hash_backward_compatible(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("hello world\n", encoding="utf-8")
    tool = FileEditTool()
    # expected_hash 未提供 → 不校验, 行为与旧版一致
    res = await tool.call(
        {"file_path": str(p), "old_string": "world", "new_string": "huginn", "working_dir": str(tmp_path)},
        _ctx(),
    )
    assert res.success is True
    assert p.read_text(encoding="utf-8") == "hello huginn\n"


# ── multi_edit_tool ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_multi_atomic_strict_aborts_all(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("aaa\n", encoding="utf-8")
    b.write_text("bbb\n", encoding="utf-8")
    tool = MultiEditTool()
    res = await tool.call({
        "edits": [
            {"file_path": str(a), "old_string": "aaa", "new_string": "AAA",
             "expected_hash": "deadbeef00000000", "hash_policy": "strict"},
            {"file_path": str(b), "old_string": "bbb", "new_string": "BBB"},
        ],
        "working_dir": str(tmp_path),
    }, _ctx())
    assert res.success is False
    assert "changed since snapshot" in res.error
    # 原子性: 两个文件都不变
    assert a.read_text(encoding="utf-8") == "aaa\n"
    assert b.read_text(encoding="utf-8") == "bbb\n"


@pytest.mark.asyncio
async def test_multi_atomic_hash_match_allows(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("aaa\n", encoding="utf-8")
    b.write_text("bbb\n", encoding="utf-8")
    tool = MultiEditTool()
    ha = _content_hash("aaa\n")
    hb = _content_hash("bbb\n")
    res = await tool.call({
        "edits": [
            {"file_path": str(a), "old_string": "aaa", "new_string": "AAA",
             "expected_hash": ha, "hash_policy": "strict"},
            {"file_path": str(b), "old_string": "bbb", "new_string": "BBB",
             "expected_hash": hb, "hash_policy": "strict"},
        ],
        "working_dir": str(tmp_path),
    }, _ctx())
    assert res.success is True
    assert a.read_text(encoding="utf-8") == "AAA\n"
    assert b.read_text(encoding="utf-8") == "BBB\n"


@pytest.mark.asyncio
async def test_multi_warn_continues(tmp_path):
    a = tmp_path / "a.txt"
    a.write_text("aaa\n", encoding="utf-8")
    tool = MultiEditTool()
    res = await tool.call({
        "edits": [
            {"file_path": str(a), "old_string": "aaa", "new_string": "AAA",
             "expected_hash": "deadbeef00000000", "hash_policy": "warn"},
        ],
        "working_dir": str(tmp_path),
    }, _ctx())
    assert res.success is True
    assert a.read_text(encoding="utf-8") == "AAA\n"


# ── 闭环: read → edit → edit ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_roundtrip_second_edit_uses_first_hash(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("hello\n", encoding="utf-8")
    tool = FileEditTool()
    # 第一次编辑: 拿到 snapshot_hash
    r1 = await tool.call(
        {"file_path": str(p), "old_string": "hello", "new_string": "hello world", "working_dir": str(tmp_path)},
        _ctx(),
    )
    assert r1.success is True
    h1 = r1.data["snapshot_hash"]
    assert h1 == _content_hash("hello\n")
    # 第二次编辑: 用第一次的 hash 作为 expected_hash → 成功
    r2 = await tool.call({
        "file_path": str(p),
        "old_string": "world",
        "new_string": "huginn",
        "expected_hash": _content_hash("hello world\n"),  # 编辑后内容 hash
        "hash_policy": "strict",
        "working_dir": str(tmp_path),
    }, _ctx())
    assert r2.success is True
    assert p.read_text(encoding="utf-8") == "hello huginn\n"
    # 但若用旧 hash (第一次快照) 做 expected_hash → 拒绝 (内容已变)
    r3 = await tool.call({
        "file_path": str(p),
        "old_string": "ginn",
        "new_string": "x",
        "expected_hash": h1,
        "hash_policy": "strict",
        "working_dir": str(tmp_path),
    }, _ctx())
    assert r3.success is False


# ── session_log 事件 ──────────────────────────────────────────────────────

def test_file_hash_mismatch_event_kind_registered():
    from huginn.events.session_log import EVENT_FILE_HASH_MISMATCH, SESSION_EVENT_KINDS
    assert EVENT_FILE_HASH_MISMATCH in SESSION_EVENT_KINDS
