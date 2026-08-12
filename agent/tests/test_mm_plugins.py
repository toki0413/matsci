"""Tests for the built-in Qwen-MM-Plugins module (dynamic resolution + modality routing)."""

from __future__ import annotations

import numpy as np

from huginn.vision.mm_plugins import (
    detect_modality,
    dynamic_resolution_hint,
    modality_routing_hint,
    recommend_resolution,
)

# ── recommend_resolution (纯函数) ─────────────────────────────────


class TestRecommendResolution:
    def test_small_image_kept(self):
        # 小图不缩放, 保留细节
        assert recommend_resolution(512, 512) == (512, 512)

    def test_large_pixels_downscaled(self):
        # 4K 图超出像素 cap → 等比缩到 cap 附近
        w, h = recommend_resolution(4000, 3000)
        assert w <= 4000 and h <= 3000
        assert w * h <= 1_200_000
        # 等比: 宽高比保持
        assert abs(w / h - 4000 / 3000) < 0.02

    def test_long_side_capped(self):
        # 超宽图 (极长边) → 最长边被 cap
        w, h = recommend_resolution(100, 8000)
        assert max(w, h) <= 2048
        assert abs(w / h - 100 / 8000) < 0.05

    def test_invalid_input(self):
        assert recommend_resolution(0, 100) == (0, 0)
        assert recommend_resolution(-5, 10) == (0, 0)

    def test_custom_cap(self):
        w, h = recommend_resolution(2000, 2000, max_pixel_cap=400_000)
        assert w * h <= 400_000


# ── detect_modality (模态路由) ────────────────────────────────────


class TestDetectModality:
    def test_filename_signal_sem(self):
        det = detect_modality("/data/sem_photo_001.png")
        assert det["modality"] == "SEM"
        assert det["action"] == "sem_analysis"
        assert det["confidence"] == "high"

    def test_filename_signal_tem(self):
        det = detect_modality("/data/hrtem_lattice.tif")
        assert det["modality"] == "TEM"
        assert det["action"] == "tem_lattice"

    def test_filename_signal_eds(self):
        det = detect_modality("/run/eds_mapping.png")
        assert det["modality"] == "EDS"
        assert det["action"] == "eds_mapping"

    def test_filename_signal_plot(self):
        det = detect_modality("/figures/xrd_plot.png")
        assert det["modality"] == "PLOT"
        assert det["action"] == "plot_extract"

    def test_bytes_no_signal(self):
        det = detect_modality(b"\x89PNG\x00\x00")
        assert det["modality"] == "UNKNOWN"
        assert det["action"] is None

    def test_missing_file_falls_back_low_conf(self):
        # 文件不存在, 无文件名信号 → UNKNOWN, 不抛异常
        det = detect_modality("/nonexistent/random_name.png")
        assert det["modality"] == "UNKNOWN"


# ── dynamic_resolution_hint / modality_routing_hint (集成) ────────


class TestHints:
    def _blank_png(self, tmp_path, w=60, h=60, name="img.png"):
        """生成一张纯色 PNG, 用于 hint 集成测试."""
        from PIL import Image

        arr = np.full((h, w, 3), 200, dtype="uint8")
        p = tmp_path / name
        Image.fromarray(arr).save(p)
        return p

    def test_dynamic_resolution_hint_small_image_ok(self, tmp_path):
        p = self._blank_png(tmp_path)
        hint = dynamic_resolution_hint(p)
        assert hint is not None
        assert "no resize" in hint

    def test_dynamic_resolution_hint_missing_file_none(self, tmp_path):
        assert dynamic_resolution_hint(tmp_path / "nope.png") is None

    def test_dynamic_resolution_hint_bytes_none(self):
        assert dynamic_resolution_hint(b"\x89PNG") is None

    def test_modality_routing_hint_plot(self, tmp_path):
        # 文件名带 plot → 推荐 plot_extract
        p = self._blank_png(tmp_path, name="bandgap_plot.png")
        hint = modality_routing_hint(p)
        assert hint is not None
        assert "[MM-Plugins]" in hint
        assert "plot_extract" in hint

    def test_modality_routing_hint_missing_none(self, tmp_path):
        assert modality_routing_hint(tmp_path / "nope.png") is None


# ── router 集成: build_cv_context 注入 MM-Plugins hint ─────────────


class TestRouterIntegration:
    def test_build_cv_context_includes_mm_plugins(self, tmp_path):
        from PIL import Image

        from huginn.vision.router import build_cv_context

        arr = np.full((80, 80, 3), 200, dtype="uint8")
        p = tmp_path / "sem_sample.png"
        Image.fromarray(arr).save(p)
        ctx = build_cv_context(str(p))
        assert "[MM-Plugins]" in ctx
        assert "sem_analysis" in ctx

    def test_coordinate_includes_mm_plugins(self, tmp_path):
        from PIL import Image

        from huginn.vision.router import VisionRouter

        arr = np.full((80, 80, 3), 200, dtype="uint8")
        p = tmp_path / "eds_overview.png"
        Image.fromarray(arr).save(p)
        router = VisionRouter()
        content, hints = router.coordinate("describe", str(p))
        assert isinstance(content, list)
        assert "[MM-Plugins]" in hints
        assert "eds_mapping" in hints
