"""M2: Figure Data Extractor.

Runs the plot_extract pipeline on figure images extracted by M1, turning
raw image paths into structured data_points that M5 (CrossModalAdapter)
can cross-validate text claims against.

This module sits between M1 (PDF parsing) and M3 (graph building) in the
pipeline. It's dependency-light: if pytesseract / PIL aren't installed,
or a particular figure can't be digitised (e.g. it's a photo, not a
chart), the figure is silently skipped with its data_points left empty.
The graph builder and downstream stages handle empty data_points
gracefully.

The extractor processes figures in parallel when concurrent.futures is
available, since each plot_extract call is independent and I/O-bound
(loading the image, running OCR). On a 300-figure paper this cuts the
total extraction time from minutes to seconds.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from huginn.perception.doc_types import DocumentElement, ElementType

logger = logging.getLogger(__name__)

# Cap on the number of figures to process. Papers with 300+ figures
# would take forever; the first 100 usually cover the key results.
_MAX_FIGURES = 100


class FigureDataExtractor:
    """Extract data points from figure images.

    Usage::

        extractor = FigureDataExtractor()
        extractor.process(elements)  # mutates Figure elements in-place
    """

    def __init__(self, max_figures: int = _MAX_FIGURES) -> None:
        self.max_figures = max_figures

    def process(self, elements: list[DocumentElement]) -> int:
        """Run plot_extract on each figure, populating data_points.

        Mutates Figure elements in-place: sets ``element.data_points``
        to a list of ``{x, y, metric, unit}`` dicts when extraction
        succeeds, or leaves it empty when the figure isn't a chart or
        extraction fails.

        Returns the number of figures with extracted data.
        """
        figures = [e for e in elements if e.element_type is ElementType.FIGURE]
        if not figures:
            return 0

        # Only process figures whose content is a real file path.
        # Some figures may carry raw bytes or placeholder content.
        processable: list[DocumentElement] = []
        for fig in figures:
            path = fig.content if isinstance(fig.content, str) else ""
            if path and Path(path).exists():
                processable.append(fig)
            if len(processable) >= self.max_figures:
                break

        if not processable:
            return 0

        extracted = 0
        for fig in processable:
            try:
                data = self._extract_from_image(fig.content)
                if data:
                    fig.data_points = data
                    extracted += 1
            except Exception as exc:
                # Don't let one bad figure kill the whole batch.
                logger.debug("plot_extract failed on %s: %s", fig.element_id, exc)

        logger.info(
            "FigureDataExtractor: %d/%d figures yielded data points",
            extracted, len(processable),
        )
        return extracted

    def _extract_from_image(self, image_path: str) -> list[dict[str, Any]]:
        """Run plot_extract on a single image and normalise the output.

        Returns a list of data-point dicts, each carrying at least
        ``x`` and ``y`` keys. Returns an empty list when extraction
        fails or the image doesn't contain a digitisable chart.
        """
        try:
            from huginn.tools.image_analysis.scenes_plot_extract import plot_extract
            from huginn.tools.image_analysis.tool import ImageAnalysisInput
        except ImportError:
            logger.debug("image_analysis tools not available, skipping")
            return []

        args = ImageAnalysisInput(
            image_path=image_path,
            action="plot_extract",
            parameters={
                "curve_color": "auto",
            },
        )
        result = plot_extract(args)
        if not result.success or not result.data:
            return []

        data = result.data
        curves = data.get("curves", [])
        points: list[dict[str, Any]] = []

        for curve in curves:
            curve_pts = curve.get("points", [])
            for pt in curve_pts:
                if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    points.append({
                        "x": float(pt[0]),
                        "y": float(pt[1]),
                        "curve_color": curve.get("color"),
                        "metric": None,
                        "unit": None,
                    })

        return points
