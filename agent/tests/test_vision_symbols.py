"""Tests for vision -> structured-symbols (symbol_encoder) and figure IR.

Covers:
- extract_chart_data: real peak detection + real (non-fabricated) intensities,
  axis calibration fallback labeling.
- visual_to_symbols_structured: confidence + self_check caveats for fallback.
- figure_ir: to_ir / ir_to_structured (pure), render smoke test.
"""

from __future__ import annotations

import numpy as np
import pytest

from huginn.vision.figure_ir import ir_to_structured, to_ir
from huginn.vision.symbol_encoder import (
    _format_symbols_structured,
    _format_symbols_text,
    extract_chart_data,
    visual_to_symbols_structured,
)


def _make_xrd_png(path, peak_cols=(40, 90, 140), size=(200, 160)):
    """合成一张带亮色峰的 XRD 风格图 (白底黑线场景简化为暗底亮峰)."""
    from PIL import Image

    arr = np.zeros((size[1], size[0]), dtype=np.uint8)
    arr[:, :] = 30  # 暗背景, 保证 axis_detected 逻辑有对比
    for c in peak_cols:
        arr[:, max(0, c - 3): c + 4] = 255
    # 人工制造一个 "主峰" 更亮, 让强度归一化有意义
    arr[:, max(0, peak_cols[1] - 3): peak_cols[1] + 4] = 200
    Image.fromarray(arr, mode="L").save(str(path))


# ── extract_chart_data ─────────────────────────────────────────


def test_extract_chart_data_finds_peaks_and_real_intensities(tmp_path):
    """应检测到峰, 且 peak_intensities 来自真实亮度 (非编造的递减假数据)."""
    img = tmp_path / "sample_xrd.png"
    _make_xrd_png(img, peak_cols=(40, 90, 140))

    data = extract_chart_data(img)
    assert "error" not in data
    assert data["image_type"] == "XRD_pattern"
    assert data["analysis_type"] == "XRD"
    assert data["n_peaks"] >= 3
    assert len(data["peak_positions_px"]) == data["n_peaks"]
    assert len(data["peak_intensities"]) == data["n_peaks"]

    # 强度必须来自图像亮度: 主峰(90, 强度200) 应高于其他峰(255 归一后=1.0)
    # 这里主峰亮度 200 < 255, 所以主峰强度应明显 < 255 峰
    by_px = dict(zip(data["peak_positions_px"], data["peak_intensities"]))
    main = by_px.get(90)
    bright = by_px.get(40)
    assert main is not None and bright is not None
    assert main < bright, "主峰亮度较低, 归一化强度应更低 — 证明强度来自真实亮度"

    # 归一化: 最亮峰应为 1.0
    assert max(data["peak_intensities"]) == pytest.approx(1.0)


def test_extract_chart_data_marks_fallback_calibration(tmp_path, monkeypatch):
    """无 OCR 可用时应标记 axis_calibration=fallback, 而非假装精确."""
    # 强制 OCR 路径不可用
    monkeypatch.setattr(
        "huginn.tools.image_analysis.scenes_plot_extract._auto_detect_axes",
        lambda *a, **k: None,
    )
    img = tmp_path / "xrd_no_ocr.png"
    _make_xrd_png(img, peak_cols=(40, 90, 140))

    data = extract_chart_data(img)
    assert data["analysis_type"] == "XRD"
    assert data["axis_calibration"].startswith("fallback")
    # 2θ 落在默认假设范围 10-90 内
    assert all(10 <= t <= 90 for t in data["estimated_2theta_deg"])


def test_extract_chart_data_uses_ocr_calibration(tmp_path, monkeypatch):
    """OCR 标定可用时 axis_calibration='ocr', 2θ 由真实轴范围映射."""
    monkeypatch.setattr(
        "huginn.tools.image_analysis.scenes_plot_extract._auto_detect_axes",
        lambda *a, **k: (0.0, 100.0, 0.0, 1000.0),  # x: 0-100, y: 0-1000
    )
    img = tmp_path / "xrd_ocr.png"
    _make_xrd_png(img, peak_cols=(40, 90, 140))
    data = extract_chart_data(img)
    assert data["axis_calibration"] == "ocr"
    # 像素 90 在轴盒 x_left=20 到 x_right=190 中 → fx=(90-20)/170≈0.412
    # 真实 x 范围 0-100 → 2θ ≈ 41 (而非 fallback 的 10+90/200*80=46)
    by_px = dict(zip(data["peak_positions_px"], data["estimated_2theta_deg"]))
    assert any(abs(t - 41) < 5 for t in by_px.values()), f"got {by_px}"


# ── visual_to_symbols_structured / self_check ──────────────────


def test_structured_self_check_marks_fallback(tmp_path, monkeypatch):
    """fallback 轴标定应进入 self_check.caveats, 让红队/PhaseGate 能判断可信度."""
    monkeypatch.setattr(
        "huginn.tools.image_analysis.scenes_plot_extract._auto_detect_axes",
        lambda *a, **k: None,
    )
    img = tmp_path / "xrd_fallback.png"
    _make_xrd_png(img, peak_cols=(40, 90, 140))

    structured = visual_to_symbols_structured(img)
    assert "error" not in structured
    assert structured["axis_calibration"].startswith("fallback")
    assert structured["self_check"]["extraction_confidence"] < 0.5
    assert any("axis_calibration" in c for c in structured["self_check"]["caveats"])


def test_format_functions_share_structured_fields(tmp_path, monkeypatch):
    """单次提取: 文本与结构化应引用同一份 chart_data (去重提取的契约)."""
    monkeypatch.setattr(
        "huginn.tools.image_analysis.scenes_plot_extract._auto_detect_axes",
        lambda *a, **k: None,
    )
    img = tmp_path / "xrd_shared.png"
    _make_xrd_png(img, peak_cols=(40, 90, 140))

    data = extract_chart_data(img)
    text = _format_symbols_text(data)
    structured = _format_symbols_structured(data)

    # 文本里的峰强度 = 结构化里的 peak_intensities (同一份数据)
    assert structured["peak_intensities"]
    assert "I≈" in text
    assert structured["axis_calibration"] in text  # fallback 标记在文本里也有


# ── figure_ir ──────────────────────────────────────────────────


def test_to_ir_builds_ir_dict():
    ir = to_ir({"Si": [1, 2, 3, 4]}, chart_type="line", title="t", style="nature")
    assert ir["chart_type"] == "line"
    assert ir["title"] == "t"
    assert ir["style"] == "nature"
    assert ir["series"][0]["label"] == "Si"


def test_ir_to_structured_roundtrip():
    ir = to_ir({"A": [1, 2], "B": [3, 4]}, chart_type="scatter", x_label="x", y_label="y")
    s = ir_to_structured(ir)
    assert s["chart_type"] == "scatter"
    assert s["n_series"] == 2
    assert s["series_labels"] == ["A", "B"]
    assert s["axes"] == {"x": "x", "y": "y"}


def test_render_scienceplots_smoke(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    ir = to_ir({"Si": [[0, 1], [1, 2], [2, 1]]}, chart_type="line")
    out = tmp_path / "fig.png"
    # render 内部会 import matplotlib; 若不可用则跳过
    try:
        from huginn.vision.figure_ir import render
    except Exception:  # pragma: no cover
        pytest.skip("figure_ir.render unavailable")
    path = render(ir, backend="scienceplots", output_path=out)
    assert out.exists()
    assert path == str(out)
