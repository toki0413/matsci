"""Tests for OCR-backed document ingestion."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest
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
