"""OCR text extraction for image files and scanned PDFs.

Tries EasyOCR first, then pytesseract, then gives up gracefully.
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
_EASYOCR_READER: Any | None = None


def _get_easyocr_reader() -> Any | None:
    """Lazy EasyOCR reader (cached at module level)."""
    global _EASYOCR_READER
    if _EASYOCR_READER is not None:
        return _EASYOCR_READER
    try:
        import easyocr

        # Suppress EasyOCR's noisy progress output.
        logging.getLogger("easyocr").setLevel(logging.WARNING)
        _EASYOCR_READER = easyocr.Reader(["en"], gpu=False, verbose=False)
        return _EASYOCR_READER
    except Exception as exc:
        logger.debug("EasyOCR not available: %s", exc)
        return None


def _ocr_image(image: Any, engine: str | None = None) -> str:
    """Run OCR on a PIL image and return extracted text."""
    from PIL import Image

    if not isinstance(image, Image.Image):
        image = Image.open(io.BytesIO(image) if isinstance(image, bytes) else image)
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")

    chosen_engine = engine or os.environ.get("HUGINN_OCR_ENGINE", "auto").lower()

    if chosen_engine in ("auto", "easyocr"):
        reader = _get_easyocr_reader()
        if reader is not None:
            try:
                import numpy as np

                results = reader.readtext(np.asarray(image), detail=0)
                return "\n".join(line.strip() for line in results if line.strip())
            except Exception as exc:
                logger.debug("EasyOCR failed: %s", exc)
                if chosen_engine == "easyocr":
                    return ""

    if chosen_engine in ("auto", "tesseract"):
        try:
            import pytesseract

            return pytesseract.image_to_string(image).strip()
        except Exception as exc:
            logger.debug("Tesseract OCR failed: %s", exc)

    return ""


def _ocr_pdf(content: bytes, engine: str | None = None) -> str:
    """Render PDF pages to images and OCR them."""
    try:
        import fitz  # pymupdf
    except Exception as exc:
        logger.debug("PyMuPDF not available for PDF OCR: %s", exc)
        return ""

    try:
        from PIL import Image
    except Exception as exc:
        logger.debug("PIL not available for PDF OCR: %s", exc)
        return ""

    parts: list[str] = []
    try:
        doc = fitz.open(stream=content, filetype="pdf")
        for page in doc:
            pix = page.get_pixmap(dpi=200)
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = _ocr_image(image, engine=engine)
            if text:
                parts.append(text)
        doc.close()
    except Exception as exc:
        logger.debug("PDF OCR failed: %s", exc)

    return "\n\n".join(parts)


def extract_text_with_ocr(filename: str, content: bytes) -> str:
    """Extract text from an image or scanned PDF using OCR.

    Returns an empty string if the file type is not supported or OCR is
    unavailable.
    """
    suffix = Path(filename).suffix.lower()
    engine = os.environ.get("HUGINN_OCR_ENGINE", "auto").lower()

    if suffix in _IMAGE_SUFFIXES:
        return _ocr_image(content, engine=engine)

    if suffix == ".pdf":
        return _ocr_pdf(content, engine=engine)

    return ""


def is_image_file(filename: str) -> bool:
    """Return True if the filename extension is a supported image format."""
    return Path(filename).suffix.lower() in _IMAGE_SUFFIXES
