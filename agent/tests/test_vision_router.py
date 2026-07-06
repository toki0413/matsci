"""Tests for the vision router module.

Covers image detection, route selection (native LLM vs CV tools vs text-only),
multimodal content building, and CV context generation with a mock encoder.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from huginn.vision.router import (
    VisionRoute,
    VisionRouter,
    build_cv_context,
    build_multimodal_content,
    detect_image_in_message,
    route_vision,
)


# ── detect_image_in_message ──────────────────────────────────────


class TestDetectImage:
    def test_plain_string_no_image(self):
        assert detect_image_in_message("What is the band gap of silicon?") is False

    def test_string_with_image_path(self):
        msg = "Please analyze /data/sem_images/sample.png for particle size"
        assert detect_image_in_message(msg) is True

    def test_string_with_jpg_path(self):
        assert detect_image_in_message("Look at C:\\images\\tem.tif") is True

    def test_dict_with_image_path_key(self):
        assert detect_image_in_message({"image_path": "/tmp/test.png"}) is True

    def test_dict_without_image_keys(self):
        assert detect_image_in_message({"text": "hello"}) is False

    def test_list_with_image_url_block(self):
        blocks = [{"type": "text", "text": "hi"}, {"type": "image_url", "image_url": {}}]
        assert detect_image_in_message(blocks) is True

    def test_none_message(self):
        assert detect_image_in_message(None) is False


# ── route_vision ────────────────────────────────────────────────


class TestRouteVision:
    def test_vision_model_with_image(self):
        assert route_vision("gpt-4o", True) == VisionRoute.NATIVE_LLM

    def test_non_vision_model_with_image(self):
        assert route_vision("deepseek-chat", True) == VisionRoute.CV_TOOLS

    def test_no_image_returns_text_only(self):
        assert route_vision("gpt-4o", False) == VisionRoute.TEXT_ONLY

    def test_no_image_with_non_vision_model(self):
        assert route_vision("deepseek-chat", False) == VisionRoute.TEXT_ONLY

    def test_unknown_model_with_image(self):
        # Unknown models return all-False caps → CV_TOOLS
        assert route_vision("totally-unknown-model-xyz", True) == VisionRoute.CV_TOOLS

    def test_claude_vision(self):
        assert route_vision("claude-3-5-sonnet-20241022", True) == VisionRoute.NATIVE_LLM


# ── build_multimodal_content ─────────────────────────────────────


class TestBuildMultimodalContent:
    def test_returns_list_with_text_and_image(self, tmp_path):
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)  # minimal PNG header
        blocks = build_multimodal_content("describe this", img)
        assert isinstance(blocks, list)
        assert len(blocks) == 2
        assert blocks[0]["type"] == "text"
        assert blocks[0]["text"] == "describe this"
        assert blocks[1]["type"] == "image_url"
        assert "base64" in blocks[1]["image_url"]["url"]

    def test_missing_file_degrades_to_text_only(self, tmp_path):
        blocks = build_multimodal_content("hello", tmp_path / "nonexistent.png")
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"

    def test_bytes_input(self):
        raw = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
        blocks = build_multimodal_content("analyze", raw)
        assert len(blocks) == 2
        assert blocks[1]["type"] == "image_url"


# ── build_cv_context ─────────────────────────────────────────────


class TestBuildCvContext:
    def test_returns_text_string(self):
        ctx = build_cv_context("/tmp/fake.png")
        assert isinstance(ctx, str)
        assert "image_analysis_tool" in ctx

    def test_with_mock_encoder_and_index(self):
        mock_encoder = MagicMock()
        mock_encoder.available = True
        mock_encoder.backend_name = "clip"

        mock_index = MagicMock()
        mock_index.search.return_value = [
            {"path": "/data/sem_001.png", "metadata": {"sample": "Si"}, "similarity": 0.92},
        ]

        ctx = build_cv_context("/tmp/test.png", mock_encoder, mock_index)
        assert "Found 1 similar" in ctx
        assert "/data/sem_001.png" in ctx
        assert "similarity=0.920" in ctx
        assert "backend=clip" in ctx

    def test_no_similar_images(self):
        mock_encoder = MagicMock()
        mock_encoder.available = False
        mock_index = MagicMock()
        mock_index.search.return_value = []

        ctx = build_cv_context("/tmp/test.png", mock_encoder, mock_index)
        assert "No similar indexed images" in ctx


# ── VisionRouter class ───────────────────────────────────────────


class TestVisionRouter:
    def test_route_text_only_when_no_image(self):
        router = VisionRouter()
        assert router.route("gpt-4o", "just a text message") == VisionRoute.TEXT_ONLY

    def test_route_native_llm_for_vision_model(self):
        router = VisionRouter()
        assert router.route("gpt-4o", "see /data/img.png") == VisionRoute.NATIVE_LLM

    def test_route_cv_tools_for_text_model(self):
        router = VisionRouter()
        assert router.route("deepseek-chat", "see /data/img.png") == VisionRoute.CV_TOOLS
