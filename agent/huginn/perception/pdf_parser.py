"""M1: PDF parser & element extractor.

Turns a PDF into a flat list of DocumentElement (text blocks, figures,
tables, captions) that the rest of the DocGraph pipeline consumes.

PyMuPDF (fitz) is the hard dependency -- it gives us text blocks with
bounding boxes, embedded images and a built-in table finder. camelot,
pdfplumber and Nougat are optional and get used when they happen to be
installed.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from huginn.perception.doc_types import BBox, DocumentElement, ElementType

logger = logging.getLogger(__name__)

# Captions in papers tend to start with one of these prefixes.
# Covers "Figure 3", "Fig. 3", "Table 2", "Tab. 2", "Scheme 1", "Chart 4"
# as well as the Chinese "图3", "表2" (with or without a space).
_CAPTION_RE = re.compile(
    r"^\s*(?:"
    r"(?:Figure|Fig|Table|Tab|Scheme|Chart)\.? *\d+"
    r"|(?:图|表|方案)\s*\d+"
    r")",
    re.IGNORECASE,
)


class PDFElementExtractor:
    """Extract structured elements from a PDF file.

    The extractor walks every page and pulls out text blocks, embedded
    images and tables, then post-processes the text blocks to flag
    captions. When Nougat is available the full-paper Markdown gets
    appended as an extra text element so downstream stages keep the
    math/structure that fitz's plain text dump loses.
    """

    def __init__(self, workspace: Path | None = None):
        self.workspace = Path(workspace).resolve() if workspace else None
        # scratch dir for extracted images -- created per extract() call
        self._fig_dir: Path | None = None
        self._fig_counter = 0
        # the currently-open document, stashed on the instance so the
        # private helpers can reach the parent doc (for extract_image)
        # without us widening their signatures.
        self._doc: Any = None
        self._pdf_path: Path | None = None

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #

    def extract(self, pdf_path: Path | str) -> list[DocumentElement]:
        """Extract all elements from a PDF file.

        Returns a flat list of DocumentElement in reading order.
        Figures and tables are interspersed with text based on
        their vertical position on the page.
        """
        try:
            import fitz  # the one hard dependency -- fail loud if missing
        except ImportError as exc:
            raise ImportError(
                "PyMuPDF (fitz) is required for PDF parsing. "
                "Install it with: pip install pymupdf"
            ) from exc

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # fresh scratch dir for this run. images live here for the rest
        # of the pipeline, so we don't clean it up ourselves -- call
        # cleanup() when you're done if you care about disk space.
        parent = self.workspace if self.workspace else None
        self._fig_dir = Path(tempfile.mkdtemp(prefix="huginn_figs_", dir=parent))
        self._fig_counter = 0

        all_elements: list[DocumentElement] = []
        doc = fitz.open(str(pdf_path))
        self._doc = doc
        self._pdf_path = pdf_path
        try:
            for page_idx, page in enumerate(doc):
                rect = page.rect
                page_w, page_h = rect.width, rect.height

                page_elements: list[DocumentElement] = []
                page_elements.extend(
                    self._extract_text_blocks(page, page_idx, page_w, page_h)
                )
                page_elements.extend(self._extract_figures(page, page_idx))
                page_elements.extend(self._extract_tables(page, page_idx))

                # reading order: top-to-bottom, left-to-right within a row.
                # fitz origin is top-left so smaller y1 == higher on page.
                page_elements.sort(key=lambda e: (e.bbox.y1, e.bbox.x1))
                all_elements.extend(page_elements)

            all_elements = self._detect_captions(all_elements)

            # Optional: fold in Nougat's structured Markdown when configured.
            nougat_md = self._try_nougat(pdf_path.read_bytes())
            if nougat_md:
                all_elements.append(
                    DocumentElement(
                        element_id="nougat_full",
                        element_type=ElementType.TEXT,
                        content=nougat_md,
                        page=0,
                        bbox=BBox(0, 0, 0, 0),
                        metadata={"source": "nougat", "full_document": True},
                    )
                )
        finally:
            with contextlib.suppress(Exception):
                doc.close()
            self._doc = None

        return all_elements

    def cleanup(self) -> None:
        """Remove the scratch directory created by the last extract() call."""
        if self._fig_dir and self._fig_dir.exists():
            import shutil

            shutil.rmtree(self._fig_dir, ignore_errors=True)
            self._fig_dir = None

    # ------------------------------------------------------------------ #
    # text blocks
    # ------------------------------------------------------------------ #

    def _extract_text_blocks(
        self, page, page_idx: int, page_w: float, page_h: float
    ) -> list[DocumentElement]:
        elements: list[DocumentElement] = []
        try:
            page_dict = page.get_text("dict")
        except Exception as exc:
            logger.debug("get_text('dict') failed on page %d: %s", page_idx, exc)
            return elements

        text_idx = 0
        for block in page_dict.get("blocks", []):
            # type 0 == text, type 1 == image block (handled by _extract_figures)
            if block.get("type", 0) != 0:
                continue
            bbox = block.get("bbox") or [0, 0, 0, 0]

            # rebuild the text line-by-line so we keep the visual line breaks
            lines: list[str] = []
            font_sizes: list[float] = []
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                line_text = "".join(span.get("text", "") for span in spans)
                for span in spans:
                    size = span.get("size")
                    if size:
                        font_sizes.append(float(size))
                if line_text.strip():
                    lines.append(line_text)

            text = "\n".join(lines).strip()
            if not text:
                continue

            avg_font = sum(font_sizes) / len(font_sizes) if font_sizes else 0.0
            elements.append(
                DocumentElement(
                    element_id=f"p{page_idx}_text_{text_idx}",
                    element_type=ElementType.TEXT,
                    content=text,
                    page=page_idx,
                    bbox=BBox(bbox[0], bbox[1], bbox[2], bbox[3]),
                    metadata={
                        "page_w": page_w,
                        "page_h": page_h,
                        "avg_font_size": round(avg_font, 2),
                    },
                )
            )
            text_idx += 1

        return elements

    # ------------------------------------------------------------------ #
    # figures
    # ------------------------------------------------------------------ #

    def _extract_figures(self, page, page_idx: int) -> list[DocumentElement]:
        elements: list[DocumentElement] = []
        if self._doc is None or self._fig_dir is None:
            return elements

        try:
            images = page.get_images(full=True)
        except Exception as exc:
            logger.debug("get_images failed on page %d: %s", page_idx, exc)
            return elements

        fig_idx = 0
        for img_info in images:
            xref = img_info[0]
            smask = img_info[1] if len(img_info) > 1 else 0

            # where the image sits on the page (it may be placed more than once)
            try:
                rects = page.get_image_rects(xref)
            except Exception:
                rects = []

            # pull the actual bytes through the parent doc
            try:
                img = self._doc.extract_image(xref)
                img_bytes = img.get("image")
                ext = img.get("ext", "png")
            except Exception as exc:
                logger.debug("extract_image xref=%s failed: %s", xref, exc)
                continue
            if not img_bytes:
                continue

            # spill to disk so downstream vision stages can mmap it
            self._fig_counter += 1
            out_name = f"p{page_idx}_fig_{fig_idx}_{self._fig_counter}.{ext}"
            out_path = self._fig_dir / out_name
            try:
                out_path.write_bytes(img_bytes)
            except Exception as exc:
                logger.debug("failed to write figure %s: %s", out_path, exc)
                continue

            if rects:
                r = rects[0]
                bbox = BBox(r.x0, r.y0, r.x1, r.y1)
            else:
                # no placement info -- span the whole page as a fallback
                bbox = BBox(0, 0, page.rect.width, page.rect.height)

            elements.append(
                DocumentElement(
                    element_id=f"p{page_idx}_fig_{fig_idx}",
                    element_type=ElementType.FIGURE,
                    content=str(out_path),
                    page=page_idx,
                    bbox=bbox,
                    raw_image=img_bytes,
                    metadata={"xref": xref, "smask": smask, "ext": ext},
                )
            )
            fig_idx += 1

        return elements

    # ------------------------------------------------------------------ #
    # tables -- try camelot, then pdfplumber, then fitz's own finder
    # ------------------------------------------------------------------ #

    def _extract_tables(self, page, page_idx: int) -> list[DocumentElement]:
        tables = self._tables_via_camelot(page_idx)
        if not tables:
            tables = self._tables_via_pdfplumber(page_idx)
        if not tables:
            tables = self._tables_via_fitz(page, page_idx)
        return tables

    def _tables_via_camelot(self, page_idx: int) -> list[DocumentElement]:
        if self._pdf_path is None:
            return []
        try:
            import camelot
        except ImportError:
            return []
        except Exception as exc:
            logger.debug("camelot import bailed: %s", exc)
            return []

        try:
            # camelot numbers pages from 1
            table_list = camelot.read_pdf(
                str(self._pdf_path), pages=str(page_idx + 1)
            )
        except Exception as exc:
            logger.debug("camelot.read_pdf failed on page %d: %s", page_idx, exc)
            return []

        # need page height to convert camelot's bottom-left origin y
        page_h = 0.0
        if self._doc is not None:
            try:
                page_h = self._doc[page_idx].rect.height
            except Exception:
                page_h = 0.0

        elements: list[DocumentElement] = []
        for i, tbl in enumerate(table_list):
            df = getattr(tbl, "df", None)
            if df is None or df.empty:
                continue
            try:
                content = df.to_markdown(index=False)
            except Exception:
                content = df.to_csv(index=False)

            bbox = self._camelot_bbox(tbl, page_h)
            elements.append(
                DocumentElement(
                    element_id=f"p{page_idx}_table_{i}",
                    element_type=ElementType.TABLE,
                    content=content,
                    page=page_idx,
                    bbox=bbox,
                    metadata={
                        "engine": "camelot",
                        "rows": int(df.shape[0]),
                        "cols": int(df.shape[1]),
                    },
                )
            )
        return elements

    @staticmethod
    def _camelot_bbox(tbl, page_h: float) -> BBox:
        """Best-effort bbox from a camelot Table.

        camelot stores coords with a bottom-left origin, so we flip y to
        match fitz's top-left origin. When anything looks off we just
        drop back to a zero bbox rather than crash.
        """
        raw = getattr(tbl, "_bbox", None)
        if not raw or len(raw) != 4 or page_h <= 0:
            return BBox(0, 0, 0, 0)
        try:
            x1, y1, x2, y2 = (float(v) for v in raw)
            top = page_h - y2
            bottom = page_h - y1
            if top > bottom:
                top, bottom = bottom, top
            return BBox(min(x1, x2), top, max(x1, x2), bottom)
        except Exception:
            return BBox(0, 0, 0, 0)

    def _tables_via_pdfplumber(self, page_idx: int) -> list[DocumentElement]:
        if self._pdf_path is None:
            return []
        try:
            import pdfplumber
        except ImportError:
            return []
        except Exception as exc:
            logger.debug("pdfplumber import bailed: %s", exc)
            return []

        elements: list[DocumentElement] = []
        try:
            with pdfplumber.open(str(self._pdf_path)) as pdf:
                if page_idx >= len(pdf.pages):
                    return []
                pg = pdf.pages[page_idx]
                found = pg.find_tables()
                for i, t in enumerate(found):
                    rows = t.extract()
                    if not rows:
                        continue
                    content = self._rows_to_markdown(rows)
                    # pdfplumber bbox: (x0, top, x1, bottom) -- already top-left
                    bb = t.bbox
                    bbox = (
                        BBox(bb[0], bb[1], bb[2], bb[3])
                        if bb and len(bb) >= 4
                        else BBox(0, 0, 0, 0)
                    )
                    elements.append(
                        DocumentElement(
                            element_id=f"p{page_idx}_table_{i}",
                            element_type=ElementType.TABLE,
                            content=content,
                            page=page_idx,
                            bbox=bbox,
                            metadata={"engine": "pdfplumber", "rows": len(rows)},
                        )
                    )
        except Exception as exc:
            logger.debug("pdfplumber failed on page %d: %s", page_idx, exc)
            return elements
        return elements

    def _tables_via_fitz(self, page, page_idx: int) -> list[DocumentElement]:
        try:
            finder = page.find_tables()
        except Exception as exc:
            logger.debug("fitz find_tables failed on page %d: %s", page_idx, exc)
            return []

        elements: list[DocumentElement] = []
        tables = getattr(finder, "tables", None) or []
        for i, t in enumerate(tables):
            try:
                rows = t.extract()
            except Exception:
                rows = []
            if not rows:
                continue
            content = self._rows_to_markdown(rows)

            raw_bbox = getattr(t, "bbox", None)
            try:
                bx = tuple(raw_bbox) if raw_bbox is not None else None
            except Exception:
                bx = None
            if bx and len(bx) >= 4:
                bbox = BBox(bx[0], bx[1], bx[2], bx[3])
            else:
                bbox = BBox(0, 0, page.rect.width, page.rect.height)

            elements.append(
                DocumentElement(
                    element_id=f"p{page_idx}_table_{i}",
                    element_type=ElementType.TABLE,
                    content=content,
                    page=page_idx,
                    bbox=bbox,
                    metadata={"engine": "fitz", "rows": len(rows)},
                )
            )
        return elements

    @staticmethod
    def _rows_to_markdown(rows: list[list[Any]]) -> str:
        """Render a 2D table (list of rows) as a markdown table."""
        if not rows:
            return ""
        cleaned = [
            ["" if c is None else str(c).replace("\n", " ").strip() for c in row]
            for row in rows
        ]
        # pad ragged rows so the markdown lines up
        width = max(len(r) for r in cleaned)
        for r in cleaned:
            while len(r) < width:
                r.append("")
        header = cleaned[0]
        body = cleaned[1:] if len(cleaned) > 1 else []
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join("---" for _ in header) + " |",
        ]
        for row in body:
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # caption detection
    # ------------------------------------------------------------------ #

    def _detect_captions(
        self, elements: list[DocumentElement]
    ) -> list[DocumentElement]:
        """Post-process: mark text blocks that look like captions."""
        for el in elements:
            if el.element_type is not ElementType.TEXT:
                continue
            if not el.content:
                continue
            if not _CAPTION_RE.match(el.content):
                continue

            el.element_type = ElementType.CAPTION
            el.metadata["is_caption"] = True

            # remember what kind of caption this is so the caption_of
            # edge builder in M4 can match it to a figure/table later
            head = el.content.lstrip().lower()
            if head.startswith("图") or head.startswith("fig") or head.startswith(
                "scheme"
            ) or head.startswith("chart"):
                el.metadata["caption_kind"] = "figure"
            elif head.startswith("表") or head.startswith("tab"):
                el.metadata["caption_kind"] = "table"
        return elements

    # ------------------------------------------------------------------ #
    # nougat (optional structured Markdown)
    # ------------------------------------------------------------------ #

    def _try_nougat(self, pdf_bytes: bytes) -> str | None:
        """Attempt Nougat for structured Markdown (optional).

        Only fires when HUGINN_DOC_ENGINE is 'nougat' or 'auto'. Returns
        None when Nougat isn't installed or produced nothing, so the
        caller can carry on with the fitz-extracted text.
        """
        engine = os.environ.get("HUGINN_DOC_ENGINE", "auto").lower()
        if engine not in ("auto", "nougat"):
            return None
        try:
            from huginn.knowledge.ocr_loader import _nougat_pdf
        except Exception as exc:
            logger.debug("can't import _nougat_pdf: %s", exc)
            return None
        try:
            md = _nougat_pdf(pdf_bytes)
        except Exception as exc:
            logger.debug("nougat run failed: %s", exc)
            return None
        if md and md.strip():
            return md
        return None
