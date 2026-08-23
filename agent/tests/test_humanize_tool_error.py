"""humanize_tool_error 的自检 — 常见裸错误要转成人话+下一步, 未知错误原样返回."""

import pytest

from huginn.tools.adapter import humanize_tool_error


@pytest.mark.parametrize(
    "raw, needle",
    [
        ("Permission denied: 'x'", "没有写权限"),
        ("FileNotFoundError: /a/b.txt", "路径不存在"),
        ("timed out after 300s", "超时"),
        ("CUDA out of memory", "内存/显存不足"),
        ("ModuleNotFoundError: No module named 'vasp'", "缺少依赖"),
    ],
)
def test_humanize_known(raw, needle):
    out = humanize_tool_error(raw)
    assert needle in out
    # 改写成"一句话+下一步", 但仍保留原文, 不丢信息
    assert "原文:" in out and raw in out


@pytest.mark.parametrize(
    "raw",
    ["", None, "some ambiguous message", "ok actually no error"],
)
def test_humanize_unknown_passthrough(raw):
    assert humanize_tool_error(raw) == raw
