"""M4: Relation Prediction Layer.

Predicts REFERENCES edges between text/mentions and the figures/tables
they point at. Two strategies, both running zero-shot at Level 1:

  1. Explicit -- "Figure 3" / "表2" mentions matched by number against
     captions. When a number maps to several candidates we disambiguate
     by spatial proximity, falling back to Hungarian assignment when
     many mentions compete for many candidates at once.

  2. Implicit -- "as shown above" / "如上图所示" phrases resolved to the
     nearest figure/table on the same or previous page. An optional LLM
     client can refine the pick; without it we rely on geometry alone.

No embeddings, no fine-tuning. The cost matrix for bipartite matching is
purely spatial at this level -- the alpha knob is wired up so Level 2
can drop in cosine similarity without touching the call sites.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from huginn.perception.doc_types import (
    DocumentElement,
    EdgeType,
    ElementType,
    GraphEdge,
)
from huginn.perception.document_graph import DocumentGraph


# Caption number patterns. Mirrors the mention regexes in document_graph.py
# but applied to caption text to recover (kind, number). Kept local so we
# don't reach into the graph builder's internals.
_CAPTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?:Figure|Fig\.?)\s*(\d+)", re.IGNORECASE), "figure"),
    (re.compile(r"(?:Table|Tab\.?)\s*(\d+)", re.IGNORECASE), "table"),
    (re.compile(r"图\s*(\d+)"), "figure"),
    (re.compile(r"表\s*(\d+)"), "table"),
]


# Implicit-reference cues. The second tuple element is the implied target
# type ("figure"/"table") or None when the phrase doesn't commit to either
# and the resolver has to search across both.
_IMPLICIT_PATTERNS: list[tuple[re.Pattern[str], str | None]] = [
    # English
    (re.compile(r"as\s+shown\s+(?:above|below|in\s+the)", re.IGNORECASE), None),
    (re.compile(r"as\s+(?:depicted|illustrated|mentioned|discussed)", re.IGNORECASE), None),
    (re.compile(r"the\s+results\s+indicate", re.IGNORECASE), None),
    (re.compile(r"see\s+(?:above|below)", re.IGNORECASE), None),
    # Chinese -- the phrase usually pins down figure vs. table
    (re.compile(r"如(?:上|下)?图(?:所示|中)?"), "figure"),
    (re.compile(r"如(?:上|下)?表(?:所示|中)?"), "table"),
    (re.compile(r"如上所述"), None),
    (re.compile(r"结果(?:表明|显示)"), None),
]


# Maps ElementType to the label our patterns use, so we can filter the
# candidate pool when an implicit phrase commits to a type.
_TYPE_LABEL: dict[ElementType, str] = {
    ElementType.FIGURE: "figure",
    ElementType.TABLE: "table",
}


# Spatial scoring knobs. Same-page pairs get a 1.5x boost; every 100 PDF
# points of bbox-center distance costs 0.1; each page of separation costs
# 0.3 (cross-page bbox distance is meaningless since coords are per-page).
_SAME_PAGE_BOOST = 1.5
_DISTANCE_PENALTY_PER_100PT = 0.1
_CROSS_PAGE_PENALTY = 0.3


class RelationPredictor:
    """Predicts REFERENCES edges and injects them into a DocumentGraph.

    Level 1: number matching + spatial heuristics, no model training.
    The ``alpha`` knob blends embedding cosine similarity with spatial
    proximity -- at Level 1 embeddings are absent so it has no effect,
    but the parameter is kept so Level 2 can plug in embeddings without
    changing any call site.
    """

    def __init__(self, alpha: float = 0.5, llm_client: Any = None):
        self.alpha = alpha
        self.llm = llm_client

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def predict(self, graph: DocumentGraph) -> list[GraphEdge]:
        """Predict REFERENCES edges and add them to the graph.

        Returns the list of newly created edges. Edges go through
        graph.add_edge(), which dedupes on (source, target, edge_type),
        so re-running predict on the same graph is a no-op.
        """
        new_edges: list[GraphEdge] = []
        new_edges.extend(self._resolve_explicit_references(graph))
        new_edges.extend(self._resolve_implicit_references(graph))
        for edge in new_edges:
            graph.add_edge(edge)
        return new_edges

    # ------------------------------------------------------------------
    # Explicit references: "Figure 3" -> figure with caption "Figure 3"
    # ------------------------------------------------------------------

    def _resolve_explicit_references(self, graph: DocumentGraph) -> list[GraphEdge]:
        """Match MENTION nodes to figures/tables by (type, number).

        Primary path: parse each CAPTION's text for a number, follow its
        CAPTION_OF edge to the figure/table it belongs to. Fallback when
        no caption carries the number: positional -- the Nth figure/table
        of the right type in reading order.
        """
        edges: list[GraphEdge] = []
        mentions = graph.get_elements(ElementType.MENTION)
        if not mentions:
            return edges

        caption_index = self._build_caption_index(graph)

        # Group mentions by (type, number) so we can spot many-to-many
        # situations and hand them to the Hungarian matcher as a batch.
        groups: dict[tuple[str, int], list[DocumentElement]] = defaultdict(list)
        for m in mentions:
            if m.mention_type and m.mention_number is not None:
                groups[(m.mention_type, m.mention_number)].append(m)

        for (kind, number), group_mentions in groups.items():
            candidates = caption_index.get((kind, number), [])

            if not candidates:
                # No caption with this number -- try positional fallback.
                # Fragile by design, only kicks in when caption parsing
                # didn't surface a match.
                target = self._positional_lookup(graph, kind, number)
                if target is not None:
                    for m in group_mentions:
                        edges.append(self._make_edge(
                            m, target, confidence=0.7,
                            match_type="explicit_positional"))
                continue

            if len(group_mentions) == 1 and len(candidates) == 1:
                # Clean 1:1 -- highest confidence.
                m = group_mentions[0]
                cap, target = candidates[0]
                edges.append(self._make_edge(
                    m, target, confidence=1.0,
                    match_type="explicit_unique", caption_id=cap.element_id))

            elif len(group_mentions) == 1:
                # One mention, several candidates -- pick the closest.
                m = group_mentions[0]
                cap, target = max(
                    candidates,
                    key=lambda ct: self._spatial_score(m, ct[1]),
                )
                edges.append(self._make_edge(
                    m, target, confidence=0.9,
                    match_type="explicit_disambiguated", caption_id=cap.element_id))

            elif len(candidates) == 1:
                # Several mentions, one candidate -- they all point at it.
                cap, target = candidates[0]
                for m in group_mentions:
                    edges.append(self._make_edge(
                        m, target, confidence=0.9,
                        match_type="explicit_shared", caption_id=cap.element_id))

            else:
                # Many-to-many -- let the Hungarian algorithm sort it out.
                targets = [t for _, t in candidates]
                matches = self._bipartite_match(group_mentions, targets, graph)
                cap_by_target = {t.element_id: c for c, t in candidates}
                for m, target in matches:
                    cap_id = cap_by_target.get(target.element_id)
                    edges.append(self._make_edge(
                        m, target, confidence=0.85,
                        match_type="explicit_bipartite", caption_id=cap_id))

        return edges

    def _build_caption_index(
        self, graph: DocumentGraph
    ) -> dict[tuple[str, int], list[tuple[DocumentElement, DocumentElement]]]:
        """Map (kind, number) -> [(caption, figure_or_table), ...].

        Follows CAPTION_OF edges to resolve each caption to its owning
        figure/table. A caption whose text we can't parse is skipped
        silently -- it just won't be a match target.
        """
        index: dict[
            tuple[str, int], list[tuple[DocumentElement, DocumentElement]]
        ] = defaultdict(list)
        for cap in graph.get_elements(ElementType.CAPTION):
            parsed = self._extract_caption_number(cap.content)
            if parsed is None:
                continue
            kind, number = parsed
            for target in graph.get_neighbors(cap.element_id, EdgeType.CAPTION_OF):
                index[(kind, number)].append((cap, target))
        return index

    def _positional_lookup(
        self, graph: DocumentGraph, kind: str, number: int
    ) -> DocumentElement | None:
        """Nth figure/table by reading order, used when no caption match.

        Reading order is page, then top-to-bottom, then left-to-right.
        Only used as a last resort -- numbering by position breaks down
        the moment a figure has no caption or captions are out of order.
        """
        target_type = ElementType.FIGURE if kind == "figure" else ElementType.TABLE
        els = sorted(
            graph.get_elements(target_type),
            key=lambda e: (e.page, e.bbox.y1, e.bbox.x1),
        )
        if 1 <= number <= len(els):
            return els[number - 1]
        return None

    # ------------------------------------------------------------------
    # Implicit references: "as shown above" -> nearest figure/table
    # ------------------------------------------------------------------

    def _resolve_implicit_references(self, graph: DocumentGraph) -> list[GraphEdge]:
        """Detect vague reference phrases and resolve them spatially.

        Source of the edge is the TEXT block containing the phrase (we
        don't synthesize a MENTION node for implicit refs at Level 1).
        Target is the nearest figure/table on the same or previous page.
        If an LLM client is wired up we ask it to confirm or override
        the spatial pick; otherwise geometry alone decides.
        """
        edges: list[GraphEdge] = []
        text_blocks = graph.get_elements(ElementType.TEXT)
        if not text_blocks:
            return edges

        figures_tables = [
            e for e in graph.get_elements()
            if e.element_type in (ElementType.FIGURE, ElementType.TABLE)
        ]
        if not figures_tables:
            return edges

        for text in text_blocks:
            if not isinstance(text.content, str):
                continue
            cue = self._detect_implicit_reference(text.content)
            if cue is None:
                continue

            implied_type, phrase = cue

            # Narrow the pool when the phrase itself says figure vs. table.
            if implied_type is not None:
                pool = [
                    e for e in figures_tables
                    if _TYPE_LABEL.get(e.element_type) == implied_type
                ]
                if not pool:
                    # Phrase said "图" but there are no figures -- fall
                    # back to searching everything rather than giving up.
                    pool = figures_tables
            else:
                pool = figures_tables

            target = self._nearest_figure_table(text, pool)
            if target is None:
                continue

            confidence = 0.6
            match_type = "implicit_spatial"

            # LLM gets a chance to override the spatial pick. If it bombs
            # or returns garbage we keep the geometry-based target.
            if self.llm is not None:
                llm_pick = self._llm_resolve(text.content, pool)
                if llm_pick is not None:
                    target = llm_pick
                    confidence = 0.7
                    match_type = "implicit_llm"

            edges.append(GraphEdge(
                source=text.element_id,
                target=target.element_id,
                edge_type=EdgeType.REFERENCES,
                confidence=confidence,
                metadata={
                    "match_type": match_type,
                    "phrase": phrase,
                    "implied_type": implied_type,
                },
            ))

        return edges

    def _detect_implicit_reference(
        self, text: str
    ) -> tuple[str | None, str] | None:
        """Return (implied_type, matched_phrase) or None.

        implied_type is "figure"/"table" when the phrase itself commits,
        None otherwise. Returns on the first hit -- a paragraph that
        says both "as shown above" and "如下图所示" is rare enough that
        we don't bother merging cues.
        """
        for pattern, kind in _IMPLICIT_PATTERNS:
            m = pattern.search(text)
            if m:
                return (kind, m.group(0))
        return None

    def _nearest_figure_table(
        self, text: DocumentElement, pool: list[DocumentElement]
    ) -> DocumentElement | None:
        """Closest figure/table on the same page, else the previous page.

        Bbox center distance is only meaningful within a page, so we
        search same-page first and only fall back to page-1 when nothing
        is co-located. Earlier pages are ignored -- "as shown above"
        almost never reaches back two pages.
        """
        same_page = [e for e in pool if e.page == text.page]
        if same_page:
            return min(same_page, key=lambda e: text.bbox.center_distance(e.bbox))
        prev_page = [e for e in pool if e.page == text.page - 1]
        if prev_page:
            return min(prev_page, key=lambda e: text.bbox.center_distance(e.bbox))
        return None

    def _llm_resolve(
        self, text_content: str, candidates: list[DocumentElement]
    ) -> DocumentElement | None:
        """Ask the LLM client to pick the referent.

        The client is expected to expose either a callable interface
        (``client(prompt) -> str``) or a ``complete(prompt) -> str``
        method. The response should be the element_id of the pick, or
        something we can fuzzy-match against candidate ids / content.
        Any exception falls back to None so the caller keeps the spatial
        pick instead of crashing.
        """
        if not candidates:
            return None
        descriptions = "\n".join(
            f"{i}: id={c.element_id} type={c.element_type.value} "
            f"page={c.page} content={str(c.content)[:120]!r}"
            for i, c in enumerate(candidates)
        )
        prompt = (
            "Pick which figure or table the following text refers to.\n"
            f"Text: {text_content[:300]}\n\n"
            f"Candidates:\n{descriptions}\n\n"
            "Reply with the element_id only."
        )
        try:
            resp = self._call_llm(prompt)
        except Exception:
            return None
        if not resp:
            return None
        resp = resp.strip()

        # Try to match the response to a candidate, going from most
        # precise to most lenient.
        for c in candidates:
            if c.element_id == resp:
                return c
        for c in candidates:
            if resp in c.element_id or (
                isinstance(c.content, str) and resp in c.content
            ):
                return c
        if resp.isdigit():
            idx = int(resp)
            if 0 <= idx < len(candidates):
                return candidates[idx]
        return None

    def _call_llm(self, prompt: str) -> str:
        """Dispatch to whichever interface the LLM client exposes."""
        if callable(self.llm):
            return str(self.llm(prompt))
        complete = getattr(self.llm, "complete", None)
        if callable(complete):
            return str(complete(prompt))
        # Unknown interface -- bail so the caller falls back to spatial.
        return ""

    # ------------------------------------------------------------------
    # Bipartite matching (Hungarian)
    # ------------------------------------------------------------------

    def _bipartite_match(
        self,
        mentions: list[DocumentElement],
        candidates: list[DocumentElement],
        graph: DocumentGraph,
    ) -> list[tuple[DocumentElement, DocumentElement]]:
        """Optimal one-to-one assignment via the Hungarian algorithm.

        Cost = 1 - similarity, where similarity blends cosine similarity
        of embeddings with spatial proximity. At Level 1 embeddings are
        absent so this collapses to pure spatial matching. Unmatched
        mentions/candidates (rectangular case) are simply dropped --
        linear_sum_assignment handles that by returning min(n, m) pairs.
        """
        n_m = len(mentions)
        n_c = len(candidates)
        if n_m == 0 or n_c == 0:
            return []
        cost = np.zeros((n_m, n_c), dtype=np.float32)
        for i, m in enumerate(mentions):
            for j, c in enumerate(candidates):
                cost[i, j] = 1.0 - self._similarity(m, c)
        row_ind, col_ind = linear_sum_assignment(cost)
        return [
            (mentions[r], candidates[c])
            for r, c in zip(row_ind, col_ind)
        ]

    def _similarity(
        self, a: DocumentElement, b: DocumentElement
    ) -> float:
        """Blend embedding cosine with spatial proximity.

        At Level 1 (no embeddings) this is just the spatial score.
        """
        spatial = self._spatial_score(a, b)
        if a.embedding is None or b.embedding is None:
            return spatial
        cos = self._cosine_sim(a.embedding, b.embedding)
        return self.alpha * cos + (1.0 - self.alpha) * spatial

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    # ------------------------------------------------------------------
    # Spatial scoring
    # ------------------------------------------------------------------

    def _spatial_score(
        self, elem1: DocumentElement, elem2: DocumentElement
    ) -> float:
        """Page proximity + bbox distance, 0..1, higher = closer.

        Same page: start at 1.0, lose 0.1 per 100pt of center distance,
        then apply the 1.5x same-page boost.
        Different page: distance is meaningless (coords are per-page), so
        penalize by 0.3 per page of separation instead.
        """
        if elem1.page == elem2.page:
            distance = elem1.bbox.center_distance(elem2.bbox)
            score = 1.0 - (distance / 100.0) * _DISTANCE_PENALTY_PER_100PT
            score *= _SAME_PAGE_BOOST
        else:
            page_diff = abs(elem1.page - elem2.page)
            score = 1.0 - page_diff * _CROSS_PAGE_PENALTY
        return max(0.0, min(1.0, score))

    # ------------------------------------------------------------------
    # Caption number extraction
    # ------------------------------------------------------------------

    def _extract_caption_number(
        self, caption_text: str
    ) -> tuple[str, int] | None:
        """Pull (kind, number) out of caption text.

        Handles "Figure 3: ...", "Fig. 3 ...", "Table 2 ...",
        "图3 ...", "表2 ..." and friends. Returns None when no number
        is found -- those captions just don't participate in matching.
        """
        if not caption_text or not isinstance(caption_text, str):
            return None
        for pattern, kind in _CAPTION_PATTERNS:
            m = pattern.search(caption_text)
            if m:
                return (kind, int(m.group(1)))
        return None

    # ------------------------------------------------------------------
    # Edge factory
    # ------------------------------------------------------------------

    @staticmethod
    def _make_edge(
        source: DocumentElement,
        target: DocumentElement,
        confidence: float,
        match_type: str,
        caption_id: str | None = None,
    ) -> GraphEdge:
        meta: dict[str, Any] = {"match_type": match_type}
        if caption_id is not None:
            meta["caption_id"] = caption_id
        return GraphEdge(
            source=source.element_id,
            target=target.element_id,
            edge_type=EdgeType.REFERENCES,
            confidence=confidence,
            metadata=meta,
        )
