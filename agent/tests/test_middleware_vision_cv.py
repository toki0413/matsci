"""Step 2 (wire-perception-channel-and-authority): middleware vision=False 转交 CV.

验证 FixDanglingToolCallsMiddleware 对 vision=False 模型不再删 image block 成
[content omitted], 而是调 build_cv_context 提取结构化视觉特征注入 [CV context]
文本通道. CV 失败/已注入时降级占位, 临时文件用完即删.
"""

from __future__ import annotations

import base64
import os
import tempfile

import pytest
from langchain_core.messages import HumanMessage

from huginn.agent.middlewares import FixDanglingToolCallsMiddleware


# deepseek-chat 在 models/registry 里是 vision=False, 触发 _strip_multimodal=True
_VISION_FALSE_MODEL = "deepseek-chat"


def _image_url_block(b64: str, mime: str = "image/png") -> dict:
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{b64}"},
    }


def _png_b64() -> str:
    # 不需要是真 PNG — build_cv_context 在测试里被 mock, 只求 base64 能解码
    return base64.b64encode(b"\x89PNG\r\n\x1a\n fake png bytes").decode("ascii")


# ── Task 2.1: vision=False + image_url → [CV context] ────────────


def test_vision_false_image_block_becomes_cv_context(monkeypatch):
    """vision=False 模型 + image_url block → 输出含 [CV context] 而非 [content omitted]."""
    monkeypatch.setattr(
        "huginn.vision.router.build_cv_context",
        lambda p: f"cv stub for {os.path.basename(p)}",
    )

    mw = FixDanglingToolCallsMiddleware(_VISION_FALSE_MODEL)
    assert mw._strip_multimodal is True, "deepseek-chat 应是 vision=False"

    msg = HumanMessage(content=[
        {"type": "text", "text": "看这张图"},
        _image_url_block(_png_b64()),
    ])
    patched = mw._patch_messages([msg])

    blocks = patched[0].content
    replaced = [b for b in blocks if b.get("type") == "text" and "[CV context]" in b["text"]]
    assert replaced, f"应产出 [CV context] text block, got {blocks}"
    assert "cv stub for" in replaced[0]["text"]
    assert "[content omitted]" not in replaced[0]["text"], "不应降级成 omitted"
    print("OK test_vision_false_image_block_becomes_cv_context")


# ── Task 2.2: 已含 [CV context] → 不重复提取 ─────────────────────


def test_cv_context_already_injected_skips_re_extraction(monkeypatch):
    """streaming.py 已注入 [CV context] 时, middleware 不再二次提取."""
    called = {"n": 0}

    def _fail_if_called(p):
        called["n"] += 1
        raise AssertionError("build_cv_context 不应被调 (已注入 [CV context])")

    monkeypatch.setattr("huginn.vision.router.build_cv_context", _fail_if_called)

    mw = FixDanglingToolCallsMiddleware(_VISION_FALSE_MODEL)

    # 另一条消息 (比如 SystemMessage 或前置 HumanMessage) 已含 [CV context] 标记
    prior = HumanMessage(content="[CV context]\n已由 streaming.py 注入的视觉描述")
    msg = HumanMessage(content=[
        {"type": "text", "text": "看这张图"},
        _image_url_block(_png_b64()),
    ])
    patched = mw._patch_messages([prior, msg])

    # image block 应退回 [content omitted], 不调 build_cv_context
    assert called["n"] == 0, "已注入 [CV context] 时不应再调 build_cv_context"
    blocks = patched[1].content
    omitted = [b for b in blocks if b.get("type") == "text" and "content omitted" in b["text"]]
    assert omitted, f"image block 应退回 [content omitted], got {blocks}"
    assert "[CV context]" not in omitted[0]["text"]
    print("OK test_cv_context_already_injected_skips_re_extraction")


# ── Task 2.3: build_cv_context 抛异常 → 降级 [content omitted] ────


def test_build_cv_context_exception_degrades_to_omitted(monkeypatch):
    """build_cv_context 抛异常 → middleware 仍返回有效 messages, image block 降级 omitted."""
    def _boom(p):
        raise RuntimeError("CV pipeline exploded")

    monkeypatch.setattr("huginn.vision.router.build_cv_context", _boom)

    mw = FixDanglingToolCallsMiddleware(_VISION_FALSE_MODEL)
    msg = HumanMessage(content=[
        {"type": "text", "text": "看这张图"},
        _image_url_block(_png_b64()),
    ])
    patched = mw._patch_messages([msg])

    # 不阻塞: 返回有效 list, image block 降级 [content omitted]
    assert isinstance(patched, list) and len(patched) == 1
    blocks = patched[0].content
    omitted = [b for b in blocks if b.get("type") == "text" and "content omitted" in b["text"]]
    assert omitted, f"CV 异常应降级 [content omitted], got {blocks}"
    assert "[CV context]" not in omitted[0]["text"]
    print("OK test_build_cv_context_exception_degrades_to_omitted")


def test_build_cv_context_empty_degrades_to_omitted(monkeypatch):
    """build_cv_context 返回空字符串 → 降级 [content omitted]."""
    monkeypatch.setattr("huginn.vision.router.build_cv_context", lambda p: "")
    mw = FixDanglingToolCallsMiddleware(_VISION_FALSE_MODEL)
    msg = HumanMessage(content=[
        {"type": "text", "text": "看这张图"},
        _image_url_block(_png_b64()),
    ])
    patched = mw._patch_messages([msg])
    blocks = patched[0].content
    omitted = [b for b in blocks if b.get("type") == "text" and "content omitted" in b["text"]]
    assert omitted, "空 CV 返回应降级 [content omitted]"
    print("OK test_build_cv_context_empty_degrades_to_omitted")


# ── Task 2.4: 临时文件清理 ────────────────────────────────────────


def test_temp_file_cleaned_up_after_cv(monkeypatch):
    """base64 解码到 NamedTemporaryFile, CV 提取完用完即删, 无残留."""
    created: list[str] = []
    _real_ntf = tempfile.NamedTemporaryFile

    def _recording_ntf(*args, **kwargs):
        f = _real_ntf(*args, **kwargs)
        created.append(f.name)
        return f

    import huginn.agent.middlewares as _mw_mod
    monkeypatch.setattr(_mw_mod.tempfile, "NamedTemporaryFile", _recording_ntf)
    monkeypatch.setattr(
        "huginn.vision.router.build_cv_context",
        lambda p: f"cv stub for {p}",
    )

    mw = FixDanglingToolCallsMiddleware(_VISION_FALSE_MODEL)
    msg = HumanMessage(content=[
        {"type": "text", "text": "看这张图"},
        _image_url_block(_png_b64()),
    ])
    mw._patch_messages([msg])

    assert created, "测试预期至少创建一个临时文件 (base64 解码路径)"
    leftover = [p for p in created if os.path.exists(p)]
    assert leftover == [], f"临时文件未清理: {leftover}"
    print("OK test_temp_file_cleaned_up_after_cv")


def test_temp_file_cleaned_even_when_cv_fails(monkeypatch):
    """CV 抛异常时临时文件仍要被删 (try/finally)."""
    created: list[str] = []
    _real_ntf = tempfile.NamedTemporaryFile

    def _recording_ntf(*args, **kwargs):
        f = _real_ntf(*args, **kwargs)
        created.append(f.name)
        return f

    import huginn.agent.middlewares as _mw_mod
    monkeypatch.setattr(_mw_mod.tempfile, "NamedTemporaryFile", _recording_ntf)
    monkeypatch.setattr(
        "huginn.vision.router.build_cv_context",
        lambda p: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    mw = FixDanglingToolCallsMiddleware(_VISION_FALSE_MODEL)
    msg = HumanMessage(content=[
        {"type": "text", "text": "看这张图"},
        _image_url_block(_png_b64()),
    ])
    mw._patch_messages([msg])

    leftover = [p for p in created if os.path.exists(p)]
    assert leftover == [], f"CV 失败时临时文件仍未清理: {leftover}"
    print("OK test_temp_file_cleaned_even_when_cv_fails")


# ── 回归: 现有行为不破坏 ──────────────────────────────────────────


def test_vision_true_keeps_image_url(monkeypatch):
    """vision=True 模型保留 image_url, 不走 CV 不删占位."""
    # gpt-4o 在 registry 里 vision=True
    mw = FixDanglingToolCallsMiddleware("gpt-4o")
    assert mw._strip_multimodal is False

    msg = HumanMessage(content=[
        {"type": "text", "text": "看这张图"},
        _image_url_block(_png_b64()),
    ])
    patched = mw._patch_messages([msg])
    blocks = patched[0].content
    # image_url block 原样保留
    img = [b for b in blocks if b.get("type") == "image_url"]
    assert img, "vision=True 应保留 image_url block"
    print("OK test_vision_true_keeps_image_url")


def test_no_image_block_zero_overhead(monkeypatch):
    """无 image block 时走原路径, 不调 build_cv_context."""
    called = {"n": 0}
    monkeypatch.setattr(
        "huginn.vision.router.build_cv_context",
        lambda p: called.__setitem__("n", called["n"] + 1) or "stub",
    )
    mw = FixDanglingToolCallsMiddleware(_VISION_FALSE_MODEL)
    # 只有一个 file block (非 image), 应走原 [content omitted] 逻辑
    msg = HumanMessage(content=[
        {"type": "text", "text": "读这个文件"},
        {"type": "file", "mime_type": "application/pdf", "base64": "abc123"},
    ])
    patched = mw._patch_messages([msg])
    assert called["n"] == 0, "无 image block 不应调 build_cv_context"
    blocks = patched[0].content
    omitted = [b for b in blocks if b.get("type") == "text" and "content omitted" in b["text"]]
    assert omitted, "file block 应走原 [content omitted] 逻辑"
    print("OK test_no_image_block_zero_overhead")


if __name__ == "__main__":
    # 不走 pytest 也能跑: 手动执行各 test_* 函数 (无 fixture 参数的)
    import sys

    class _MP:
        def setattr(self, *a, **k):
            # 简化: 直接用 monkeypatch 替换太重, 这里只跑不需 mock 的回归用例
            pass

    # 单独跑 vision=True 回归 (不需 mock)
    test_vision_true_keeps_image_url(_MP())
    print("[middleware_vision_cv] regression self-check OK")
    sys.exit(0)
