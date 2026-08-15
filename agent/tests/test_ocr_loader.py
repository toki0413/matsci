"""Tests for OCR-backed document ingestion."""

from __future__ import annotations

import pytest

pytest.importorskip("PIL")

import io
from pathlib import Path
from typing import Any

from PIL import Image

from huginn.knowledge import ocr_loader
from huginn.knowledge.ocr_loader import extract_text_with_ocr, is_image_file
from huginn.knowledge.store import _extract_text
from huginn.rag.vector_store import VectorStore


def _make_image_bytes(text_marker: str = "HUGINN") -> bytes:
    """Create a tiny dummy image as bytes."""
    img = Image.new("RGB", (200, 50), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _patch_easyocr_reader(monkeypatch: Any, lines: list[str]) -> None:
    """Replace EasyOCR Reader with a fake that returns the given lines."""

    class FakeReader:
        def readtext(self, image: Any, detail: int = 0) -> list[str]:
            return lines

    monkeypatch.setattr(ocr_loader, "_EASYOCR_READER", FakeReader(), raising=False)


class TestOcrLoader:
    def test_is_image_file(self) -> None:
        assert is_image_file("scan.png")
        assert is_image_file("photo.JPG")
        assert not is_image_file("doc.pdf")
        assert not is_image_file("notes.md")

    def test_extract_text_from_image_uses_easyocr(self, monkeypatch: Any) -> None:
        _patch_easyocr_reader(monkeypatch, ["Hello", "from OCR"])
        content = _make_image_bytes()
        text = extract_text_with_ocr("scan.png", content)
        assert "Hello" in text
        assert "from OCR" in text

    def test_extract_text_from_unsupported_returns_empty(self) -> None:
        assert extract_text_with_ocr("notes.md", b"# hello") == ""

    def test_extract_text_obeys_engine_env(self, monkeypatch: Any) -> None:
        """If HUGINN_OCR_ENGINE=tesseract and tesseract fails, return empty."""
        monkeypatch.setenv("HUGINN_OCR_ENGINE", "tesseract")
        monkeypatch.setattr(ocr_loader, "_EASYOCR_READER", None, raising=False)
        content = _make_image_bytes()
        # pytesseract is not installed in this environment, so it should fail.
        text = extract_text_with_ocr("scan.png", content)
        assert text == ""


class TestKnowledgeBaseOcrIntegration:
    def test_extract_text_routes_images_to_ocr(self, monkeypatch: Any) -> None:
        _patch_easyocr_reader(monkeypatch, ["OCR text from image"])
        content = _make_image_bytes()
        text = _extract_text("scan.png", content)
        assert "OCR text from image" in text

    def test_extract_text_pdf_uses_ocr_when_text_empty(
        self, monkeypatch: Any, tmp_path: Path
    ) -> None:
        _patch_easyocr_reader(monkeypatch, ["scanned page"])
        # Create a minimal blank PDF using pymupdf.
        try:
            import fitz
        except ImportError:
            pytest.skip("pymupdf not installed")

        pdf_path = tmp_path / "blank.pdf"
        doc = fitz.open()
        page = doc.new_page(width=200, height=200)
        # Draw a white rectangle so the page is not completely empty.
        page.draw_rect(fitz.Rect(0, 0, 200, 200), color=(1, 1, 1), fill=(1, 1, 1))
        doc.save(str(pdf_path))
        doc.close()

        content = pdf_path.read_bytes()
        text = _extract_text("blank.pdf", content)
        assert "scanned page" in text


class TestVectorStoreOcrIntegration:
    def test_parse_file_routes_images_to_ocr(
        self, monkeypatch: Any, tmp_path: Path
    ) -> None:
        _patch_easyocr_reader(monkeypatch, ["vector store OCR"])
        img_path = tmp_path / "scan.png"
        img_path.write_bytes(_make_image_bytes())

        vs = VectorStore(persist_dir=str(tmp_path / "rag"))
        text = vs._parse_file(img_path)
        assert "vector store OCR" in text

    def test_parse_file_pdf_uses_ocr_when_no_text(
        self, monkeypatch: Any, tmp_path: Path
    ) -> None:
        _patch_easyocr_reader(monkeypatch, ["pdf ocr fallback"])
        try:
            import fitz
        except ImportError:
            pytest.skip("pymupdf not installed")

        pdf_path = tmp_path / "blank.pdf"
        doc = fitz.open()
        page = doc.new_page(width=200, height=200)
        page.draw_rect(fitz.Rect(0, 0, 200, 200), color=(1, 1, 1), fill=(1, 1, 1))
        doc.save(str(pdf_path))
        doc.close()

        vs = VectorStore(persist_dir=str(tmp_path / "rag"))
        text = vs._parse_file(pdf_path)
        assert "pdf ocr fallback" in text


class TestLlmVisionCallback:
    """新增: LLM-as-OCR 可用性查询 (HUGINN 视觉压缩门控)."""

    def test_llm_vision_available_offers_guard(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(ocr_loader, "_LLM_VISION_CALLBACK", None, raising=False)
        assert ocr_loader.llm_vision_available() is False

    def test_llm_vision_available_true_after_set(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(
            ocr_loader, "_LLM_VISION_CALLBACK", lambda b, h: "", raising=False
        )
        assert ocr_loader.llm_vision_available() is True


class TestEmbeddingAndTokenize:
    """新增: embedding 维度动态推导 + BM25 jieba 分词 (知识库升级)."""

    def test_embed_model_env_override(self, monkeypatch: Any) -> None:
        monkeypatch.setenv("HUGINN_EMBED_MODEL", "custom-model")
        import importlib

        import huginn.knowledge.store as store

        reloaded = importlib.reload(store)
        assert reloaded.EMBED_MODEL == "custom-model"
        monkeypatch.delenv("HUGINN_EMBED_MODEL")
        importlib.reload(store)

    def test_deterministic_vectors_dim_dynamic(self) -> None:
        import numpy as np

        from huginn.knowledge.store import _deterministic_vectors

        vecs = _deterministic_vectors(["材料科学", "催化"], dim=384)
        assert vecs.shape == (2, 384)
        # 归一化后范数为 1
        norms = np.linalg.norm(vecs, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-3)

    def test_tokenize_uses_jieba_when_available(self, monkeypatch: Any) -> None:
        import huginn.knowledge.store as store

        # 强制走 jieba 路径
        monkeypatch.setattr(store, "_JIEBA", None, raising=False)
        if store._get_jieba() is None:
            pytest.skip("jieba not installed")
        tokens = store._tokenize("高熵合金 的 fatigue")
        # jieba 应切出 "合金" 而非按字拆成 "合","金"
        assert "合金" in tokens or "高熵合金" in tokens

    def test_tokenize_fallback_without_jieba(self, monkeypatch: Any) -> None:
        import huginn.knowledge.store as store

        monkeypatch.setattr(store, "_get_jieba", lambda: None)
        tokens = store._tokenize("高熵合金 fatigued")
        assert "fatigued" in tokens
        # 中文按字切 (无 jieba 时)
        assert "合" in tokens and "金" in tokens
