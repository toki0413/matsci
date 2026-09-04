"""Tests for long pasted-user-text auto-offload (paste-to-file)."""

from __future__ import annotations

import os

import pytest

from huginn.tools.paste_offload import maybe_offload_pasted_text

LONG_TEXT = "这是超长粘贴内容。\n" * 3000  # ~4.2w 字符


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    """把 offload 目录指到临时目录, 避免污染真实 runtime home."""
    monkeypatch.setenv("HUGINN_CACHE_DIR", str(tmp_path))
    return tmp_path / "tool_artifacts"


def test_long_text_offloaded(cache_dir):
    """超长文本应被落盘, 消息替换成预览, 不内联进 prompt."""
    preview, path = maybe_offload_pasted_text(LONG_TEXT, threshold_chars=20000)
    assert path is not None
    assert preview != LONG_TEXT
    # 预览包含路径与字符数, 并提示用 file_read_tool 读取
    assert preview.startswith("[已保存超长文本到 ")
    assert str(path) in preview
    assert f"{len(LONG_TEXT)}字符" in preview
    assert "file_read_tool" in preview
    # 文件确实写到了 offload 目录
    assert os.path.exists(path)
    assert path.startswith(str(cache_dir))
    with open(path, encoding="utf-8") as f:
        assert f.read() == LONG_TEXT


def test_below_threshold_unchanged(cache_dir):
    """低于阈值时原样返回, 不做任何改动, 也不落盘."""
    short = "普通短消息，不该被落盘"
    preview, path = maybe_offload_pasted_text(short, threshold_chars=20000)
    assert preview == short
    assert path is None
    assert list(cache_dir.glob("*.txt")) == []


def test_disabled_param_unchanged(cache_dir):
    """enabled=False 时即使超长也不落盘."""
    preview, path = maybe_offload_pasted_text(
        LONG_TEXT, threshold_chars=20000, enabled=False
    )
    assert preview == LONG_TEXT
    assert path is None


def test_env_disable(monkeypatch, tmp_path):
    """HUGINN_PASTE_OFFLOAD=0 关闭开关, 超长文本原样返回."""
    monkeypatch.setenv("HUGINN_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("HUGINN_PASTE_OFFLOAD", "0")
    preview, path = maybe_offload_pasted_text(LONG_TEXT)
    assert preview == LONG_TEXT
    assert path is None


def test_env_threshold(monkeypatch, tmp_path):
    """通过 env 调低阈值, 使较短的文本也能触发落盘."""
    monkeypatch.setenv("HUGINN_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("HUGINN_PASTE_OFFLOAD_THRESHOLD", "100")
    text = "x" * 200
    preview, path = maybe_offload_pasted_text(text)
    assert path is not None
    assert f"{len(text)}字符" in preview
    with open(path, encoding="utf-8") as f:
        assert f.read() == text


def test_empty_and_non_text_unchanged(cache_dir):
    """空串 / 非文本原样返回, 不落盘."""
    assert maybe_offload_pasted_text("", threshold_chars=1) == ("", None)
    # 非字符串类型也不应抛错
    assert maybe_offload_pasted_text(None, threshold_chars=1) == (None, None)  # type: ignore[arg-type]
