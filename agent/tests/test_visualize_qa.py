"""visualize_qa 纯函数单测 — 用 PIL 合成图验证每个 QA flag.

覆盖: blank / clipped / extremely_elongated / pass / error (缺失文件 & 非图片).
不依赖 LLM, 运行快, 不触碰全局状态.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from huginn.tools.visualize_qa import (
    VERDICT_ERROR,
    VERDICT_FAIL,
    VERDICT_FIX,
    VERDICT_PASS,
    check_aspect,
    check_blank,
    check_clipped,
    check_cluttered,
    qa_directive,
    qa_figure,
)


def _save(img: Image.Image, path: Path) -> Path:
    img.save(path)
    return path


def _solid(size=(200, 150), value: int = 255) -> Image.Image:
    return Image.new("L", size, value)


def _figure_like(size=(200, 150)):
    """白底 + 中央黑色矩形(不贴边) — 模拟正常图表."""
    img = _solid(size)
    d = ImageDraw.Draw(img)
    d.rectangle([40, 30, 160, 120], fill=0)
    return img


def test_blank_solid_white_fails(tmp_path: Path) -> None:
    p = _save(_solid(), tmp_path / "blank.png")
    r = qa_figure(p)
    assert r["verdict"] == VERDICT_FAIL
    assert "blank" in r["flags"]
    assert "clipped" not in r["flags"]


def test_blank_empty_std(tmp_path: Path) -> None:
    # 灰底纯色: std=0 → blank
    p = _save(_solid(value=128), tmp_path / "flat.png")
    flag, m = check_blank(np.asarray(Image.open(p), dtype=float))
    assert flag == "blank"
    assert m["std"] == 0.0


def test_normal_figure_passes(tmp_path: Path) -> None:
    p = _save(_figure_like(), tmp_path / "ok.png")
    r = qa_figure(p)
    assert r["verdict"] == VERDICT_PASS, r
    assert r["flags"] == []
    assert qa_directive(r) == ""


def test_clipped_content_touches_edge(tmp_path: Path) -> None:
    # 横贯左右边缘的横条 → clipped (但保留白底, 不是 blank)
    img = _solid()
    ImageDraw.Draw(img).rectangle([0, 60, 199, 90], fill=0)
    p = _save(img, tmp_path / "clipped.png")
    r = qa_figure(p)
    assert "clipped" in r["flags"]
    assert r["verdict"] == VERDICT_FAIL


def test_clipped_flag_function(tmp_path: Path) -> None:
    img = _solid()
    ImageDraw.Draw(img).rectangle([0, 60, 199, 90], fill=0)  # 一条横贯左右的线
    p = _save(img, tmp_path / "vline.png")
    flag, m = check_clipped(np.asarray(Image.open(p), dtype=float))
    assert flag == "clipped"
    assert m["edge_content_ratio"] > 0


def test_extremely_elongated_fix(tmp_path: Path) -> None:
    # 100x1000 竖长条 → fix_needed
    img = _solid((100, 1000))
    ImageDraw.Draw(img).rectangle([20, 200, 80, 800], fill=0)
    p = _save(img, tmp_path / "tall.png")
    r = qa_figure(p)
    assert "extremely_elongated" in r["flags"]
    assert r["verdict"] == VERDICT_FIX
    assert qa_directive(r)  # 非空


def test_aspect_function() -> None:
    flag, m = check_aspect(np.zeros((1000, 100), dtype=float))
    assert flag == "extremely_elongated"
    assert m["aspect"] == pytest.approx(0.1)
    flag2, _ = check_aspect(np.zeros((100, 200), dtype=float))
    assert flag2 is None


def test_cluttered_high_edge_density() -> None:
    # 布满随机黑白噪点 → 内部边缘密度高 → cluttered
    rng = np.random.default_rng(0)
    noisy = (rng.random((200, 200)) > 0.5).astype(float) * 255.0
    flag, m = check_cluttered(noisy)
    assert flag == "cluttered"
    assert m["inner_edge_density"] > 0.25


def test_cluttered_too_small_image() -> None:
    # h<8 或 w<8 → 直接返回 None + note
    flag, m = check_cluttered(np.zeros((5, 200), dtype=float))
    assert flag is None
    assert m["note"] == "too small"
    assert m["inner_edge_density"] == 0.0


def test_cluttered_numpy_gradient_fallback(monkeypatch) -> None:
    # scipy.sobel 缺失/抛错 → numpy.gradient 兜底梯度
    import scipy.ndimage

    def _boom(*a, **k):
        raise ImportError("no scipy")

    monkeypatch.setattr(scipy.ndimage, "sobel", _boom)
    rng = np.random.default_rng(0)
    noisy = (rng.random((200, 200)) > 0.5).astype(float) * 255.0
    flag, m = check_cluttered(noisy)
    assert flag == "cluttered"
    assert m["inner_edge_density"] > 0.25


def test_qa_check_exception_does_not_block(monkeypatch, tmp_path: Path) -> None:
    # 某个检查抛异常 → 记录 metrics 错误并 continue, 不阻断整体
    import huginn.tools.visualize_qa as vq

    def _boom(gray):
        raise RuntimeError("aspect boom")

    _boom.__name__ = "check_aspect"
    monkeypatch.setattr(vq, "check_aspect", _boom)
    p = _save(_figure_like(), tmp_path / "ok.png")
    r = qa_figure(p)
    assert r["metrics"]["check_aspect"]["error"] == "aspect boom"
    assert r["verdict"] == VERDICT_PASS  # 其余检查通过, 整体仍 pass


def test_missing_file_error(tmp_path: Path) -> None:
    r = qa_figure(tmp_path / "nope.png")
    assert r["verdict"] == VERDICT_ERROR
    assert "file not found" in r["error"]
    assert qa_directive(r).startswith("visual QA error")


def test_non_image_file_error(tmp_path: Path) -> None:
    p = tmp_path / "notimg.png"
    p.write_bytes(b"this is not a png")
    r = qa_figure(p)
    assert r["verdict"] == VERDICT_ERROR


def test_dynamic_resolution_downscales_large_image(tmp_path: Path) -> None:
    # 4000px 宽大图 → 等比缩到 max_dim=2048, 判定仍正常
    big = _solid((4000, 2000))
    ImageDraw.Draw(big).rectangle([500, 300, 3500, 1700], fill=0)
    p = _save(big, tmp_path / "big.png")
    r = qa_figure(p, max_dim=2048)
    img_meta = r["metrics"]["image"]
    assert img_meta["downscaled"] is True
    assert img_meta["width"] <= 2048
    assert img_meta["height"] <= 2048
    assert r["verdict"] == VERDICT_PASS, r
