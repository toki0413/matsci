"""Shared data types for the document understanding pipeline.

These types flow through every module in the DocGraph system:
  PDF → M1(elements) → M3(graph) → M4(relations) → M5(validation) → M6(packages)

Keeping them in one place avoids circular imports between the parser,
graph builder, and downstream consumers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class ElementType(str, Enum):
    """What kind of document element this is."""

    TEXT = "text"
    FIGURE = "figure"
    TABLE = "table"
    CAPTION = "caption"
    MENTION = "mention"          # "Figure 3", "Table 2"
    FORMULA = "formula"
    DATA_POINT = "data_point"    # extracted from chart
    CLAIM = "claim"              # quantitative/qualitative assertion


class EdgeType(str, Enum):
    """Typed edges in the heterogeneous document graph."""

    SEQ = "seq"                  # reading-order adjacency (text→text)
    CONTAINS = "contains"        # text block contains a mention
    CAPTION_OF = "caption_of"    # caption belongs to figure/table
    ADJACENT = "adjacent"        # spatial bbox proximity
    EXTRACTED_FROM = "extracted_from"  # data points from figure
    REFERENCES = "references"     # mention → figure/table (predicted)
    SUPPORTS = "supports"        # claim → data (validated, positive)
    CONTRADICTS = "contradicts"   # claim → data (validated, negative)
    INCONCLUSIVE = "inconclusive"  # claim → data (ambiguous)


@dataclass
class BBox:
    """Bounding box in PDF coordinates (origin top-left, points)."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def normalized(self, page_w: float, page_h: float) -> tuple[float, float, float, float]:
        """Return bbox normalized to [0, 1] for position encoding."""
        return (
            self.x1 / page_w,
            self.y1 / page_h,
            self.x2 / page_w,
            self.y2 / page_h,
        )

    def iou(self, other: BBox) -> float:
        """Intersection-over-union with another bbox."""
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def center_distance(self, other: BBox) -> float:
        """Euclidean distance between centers."""
        cx1, cy1 = (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2
        cx2, cy2 = (other.x1 + other.x2) / 2, (other.y1 + other.y2) / 2
        return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5


@dataclass
class DocumentElement:
    """A single element extracted from a PDF.

    This is the atomic unit that flows through the entire pipeline.
    """

    element_id: str
    element_type: ElementType
    content: str                    # text content or image path
    page: int                       # 0-indexed page number
    bbox: BBox
    raw_image: bytes | None = None  # for figure/table: the extracted image bytes
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: np.ndarray | None = None    # unified representation space vector

    # Fields populated by downstream modules

    mention_type: str | None = None        # "figure" / "table" (for MENTION type)
    mention_number: int | None = None      # 3 for "Figure 3"
    claim_data: dict[str, Any] | None = None  # structured claim (for CLAIM type)
    data_points: list[dict] | None = None   # extracted numerical data (for FIGURE type)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict for API responses / JSON storage."""
        d = {
            "element_id": self.element_id,
            "element_type": self.element_type.value,
            "content": self.content[:500] if isinstance(self.content, str) else str(self.content)[:500],
            "page": self.page,
            "bbox": [self.bbox.x1, self.bbox.y1, self.bbox.x2, self.bbox.y2],
            "metadata": self.metadata,
        }
        if self.mention_type:
            d["mention_type"] = self.mention_type
            d["mention_number"] = self.mention_number
        if self.claim_data:
            d["claim_data"] = self.claim_data
        if self.data_points:
            d["data_points"] = self.data_points[:20]  # cap for API
        return d


@dataclass
class GraphEdge:
    """A typed edge between two document elements."""

    source: str          # element_id
    target: str          # element_id
    edge_type: EdgeType
    weight: float = 1.0
    confidence: float = 1.0     # for predicted edges
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InformationPackage:
    """A bundled output unit: text + data + validation, traceable to source.

    This is the final deliverable of the pipeline. Each package bundles
    related elements into a coherent unit that a downstream LLM can
    consume as a single context block.
    """

    package_id: str
    elements: list[DocumentElement] = field(default_factory=list)
    text_blocks: list[str] = field(default_factory=list)
    figures: list[str] = field(default_factory=list)     # image paths
    data_points: list[dict] = field(default_factory=list)
    claims: list[dict] = field(default_factory=list)
    validation_results: list[dict] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "package_id": self.package_id,
            "n_elements": len(self.elements),
            "text_blocks": self.text_blocks[:5],
            "figures": self.figures,
            "data_points": self.data_points[:10],
            "claims": self.claims,
            "validation_results": self.validation_results,
            "summary": self.summary,
        }
