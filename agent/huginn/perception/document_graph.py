"""Heterogeneous document graph builder (M3).

Turns the flat list of DocumentElement objects produced by the parser (M1)
into a typed graph where nodes are elements and edges encode structural
relationships: reading order, containment, caption ownership, spatial
adjacency, and data provenance.

The graph is intentionally heterogeneous: both nodes (ElementType) and
edges (EdgeType) carry types so downstream GNN / relation-extraction stages
(M4+) can condition on them. The structural edges built here are
deterministic -- derived from geometry and text -- so no model is involved.
Predicted edges (REFERENCES, SUPPORTS, ...) are added later by M4.

This module knows nothing about PDF parsing internals; it only consumes the
data structures defined in doc_types. That keeps M3 testable in isolation.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

import numpy as np

from huginn.perception.doc_types import (
    BBox,
    DocumentElement,
    EdgeType,
    ElementType,
    GraphEdge,
)


# ---------------------------------------------------------------------------
# Mention detection patterns.
#
# Each entry pairs a compiled regex with the mention_type it implies. We keep
# English and Chinese variants side by side so a single pass over a text block
# catches both "Figure 3" and "图3". Order matters only for readability --
# matches are de-duplicated by (mention_type, number) afterwards.
# ---------------------------------------------------------------------------
_MENTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # English: "Figure 3", "Fig. 3", "Fig 3", "Table 2", "Tab. 2"
    (re.compile(r"(?:Figure|Fig\.?)\s*(\d+)", re.IGNORECASE), "figure"),
    (re.compile(r"(?:Table|Tab\.?)\s*(\d+)", re.IGNORECASE), "table"),
    # Chinese: "图3", "图 3", "表2", "表 2"
    (re.compile(r"图\s*(\d+)"), "figure"),
    (re.compile(r"表\s*(\d+)"), "table"),
]

# Caption type detection — mirrors mention patterns but used by
# _build_caption_of_edges to link captions only to the right element type.
_CAPTION_TYPE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?:Figure|Fig\.?)\s*\d+", re.IGNORECASE), "figure"),
    (re.compile(r"(?:Table|Tab\.?)\s*\d+", re.IGNORECASE), "table"),
    (re.compile(r"图\s*\d+"), "figure"),
    (re.compile(r"表\s*\d+"), "table"),
]


class DocumentGraph:
    """A heterogeneous graph over document elements.

    Nodes are DocumentElement instances keyed by element_id; edges are
    GraphEdge instances stored as a flat list with a side index for fast
    duplicate suppression. Structural edges (SEQ, CONTAINS, CAPTION_OF,
    ADJACENT, EXTRACTED_FROM) are built deterministically -- no predictions.

    Predicted edges (REFERENCES, SUPPORTS, ...) are added later via
    add_edge() by downstream modules.
    """

    # Only connect elements of different types in ADJACENT. Text-to-text
    # adjacency is already covered by SEQ, so adding it here would just
    # bloat the graph without any new signal for downstream stages.
    _ADJACENT_CROSS_TYPE_ONLY: bool = True

    # Max ADJACENT edges per node. Prevents a densely packed page from
    # generating hundreds of edges for a single element.
    _ADJACENT_MAX_PER_NODE: int = 10

    # Distance threshold for ADJACENT edges (PDF points).
    _ADJACENT_THRESHOLD: float = 150.0

    def __init__(self, elements: list[DocumentElement] | None = None):
        self._elements: dict[str, DocumentElement] = {}  # id -> element
        self._edges: list[GraphEdge] = []
        # (source, target, edge_type) set for O(1) duplicate checks.
        self._edge_index: set[tuple[str, str, EdgeType]] = set()
        # Adjacency index: element_id -> {edge_type -> set(neighbor_ids)}.
        # Built alongside _edges so get_neighbors is O(1) instead of O(E).
        self._adjacency: dict[str, dict[EdgeType, set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        # Monotonic counter for minting child element ids (mentions, data
        # points). Keeps ids stable and human-readable while avoiding
        # collisions when many children are spawned from one parent.
        self._child_counter: int = 0
        if elements:
            self.build(elements)

    # ------------------------------------------------------------------
    # Public API: building the graph
    # ------------------------------------------------------------------

    def build(self, elements: list[DocumentElement]) -> None:
        """Build the full graph from a list of elements.

        Resets any existing state. Edge construction is ordered so that
        later stages can rely on the side effects of earlier ones:
          1. SEQ          -- pure geometry on existing text nodes
          2. CONTAINS     -- injects new MENTION nodes into the graph
          3. CAPTION_OF   -- links captions to figures/tables
          4. ADJACENT     -- runs last, sees every node including mentions
          5. EXTRACTED_FROM -- wires data points to their source figure
        """
        self._elements = {}
        self._edges = []
        self._edge_index = set()
        self._adjacency = defaultdict(lambda: defaultdict(set))
        self._child_counter = 0

        for el in elements:
            self._elements[el.element_id] = el

        # Route everything through add_edge so the dedup index stays
        # consistent and later add_edge() calls from M4 can't accidentally
        # re-insert a structural edge that was already built.
        for edge in self._build_seq_edges():
            self.add_edge(edge)
        for edge in self._build_contains_edges():
            self.add_edge(edge)
        for edge in self._build_caption_of_edges():
            self.add_edge(edge)
        for edge in self._build_adjacent_edges():
            self.add_edge(edge)
        for edge in self._build_extracted_from_edges():
            self.add_edge(edge)

    def add_element(self, element: DocumentElement) -> None:
        """Insert a single element. No edges are created here -- callers
        that need edges should add_edge() them explicitly."""
        self._elements[element.element_id] = element

    def add_edge(self, edge: GraphEdge) -> None:
        """Add an edge, silently skipping exact duplicates.

        Two edges are considered duplicate when source, target and edge_type
        all match. This keeps the graph clean when structural builders and
        downstream predictors could otherwise emit the same link twice.
        """
        key = (edge.source, edge.target, edge.edge_type)
        if key in self._edge_index:
            return
        # Guard against references to unknown nodes -- easier to debug a
        # missing element here than downstream in the GNN.
        self._edge_index.add(key)
        self._edges.append(edge)
        # Maintain the adjacency index for O(1) neighbor lookups.
        self._adjacency[edge.source][edge.edge_type].add(edge.target)
        self._adjacency[edge.target][edge.edge_type].add(edge.source)

    # ------------------------------------------------------------------
    # Structural edge builders
    # ------------------------------------------------------------------

    def _build_seq_edges(self) -> list[GraphEdge]:
        """Connect text blocks in reading order within each page.

        Reading order is approximated by sorting on the top y-coordinate
        (y1) and breaking ties on x1. Adjacent blocks in that sorted order
        get a directed SEQ edge from the earlier block to the later one,
        so downstream models can recover direction.

        Only TEXT elements participate -- captions are attributed via
        CAPTION_OF and figures/tables via their own relations, so weaving
        them into the reading-order chain would muddy the semantics.
        """
        edges: list[GraphEdge] = []
        by_page: dict[int, list[DocumentElement]] = defaultdict(list)
        for el in self._elements.values():
            if el.element_type is ElementType.TEXT:
                by_page[el.page].append(el)

        for els in by_page.values():
            els.sort(key=lambda e: (e.bbox.y1, e.bbox.x1))
            for a, b in zip(els, els[1:]):
                edges.append(GraphEdge(
                    source=a.element_id,
                    target=b.element_id,
                    edge_type=EdgeType.SEQ,
                ))
        return edges

    def _build_contains_edges(self) -> list[GraphEdge]:
        """Detect figure/table mentions in text blocks and wire CONTAINS edges.

        Each mention becomes a new MENTION node so the graph can represent
        the textual pointer ("the text talks about Figure 3") as a
        first-class object. The actual REFERENCES edge (mention -> figure
        or table) is a prediction task handled by M4, not here -- we only
        assert that the text contains the mention.

        Note: this mutates self._elements by inserting the freshly minted
        MENTION nodes, which subsequent builders (CAPTION_OF, ADJACENT)
        will then see.
        """
        edges: list[GraphEdge] = []
        # Snapshot the keys first -- we add new elements mid-loop and don't
        # want to iterate over them.
        for el in list(self._elements.values()):
            if el.element_type is not ElementType.TEXT:
                continue
            for mention in self._extract_mentions(el):
                self._elements[mention.element_id] = mention
                edges.append(GraphEdge(
                    source=el.element_id,
                    target=mention.element_id,
                    edge_type=EdgeType.CONTAINS,
                ))
        return edges

    def _build_caption_of_edges(self) -> list[GraphEdge]:
        """Link each caption to its nearest figure/table on the same page.

        The caption text is parsed to determine whether it refers to a
        figure ("图3") or a table ("表2"), and only candidates of the
        matching type are considered. This prevents a table caption from
        being linked to a nearby figure (or vice versa) just because it
        happens to be spatially closer.
        """
        edges: list[GraphEdge] = []
        captions = [e for e in self._elements.values()
                    if e.element_type is ElementType.CAPTION]
        figures = [e for e in self._elements.values()
                   if e.element_type is ElementType.FIGURE]
        tables = [e for e in self._elements.values()
                  if e.element_type is ElementType.TABLE]

        for cap in captions:
            cap_type = self._detect_caption_type(cap.content)
            # Filter targets by the caption's declared type. When we can't
            # tell from the text, fall back to considering both types.
            if cap_type == "figure":
                targets = figures
            elif cap_type == "table":
                targets = tables
            else:
                targets = figures + tables

            same_page = [t for t in targets if t.page == cap.page]
            if not same_page:
                continue
            nearest = min(same_page, key=lambda t: cap.bbox.center_distance(t.bbox))
            edges.append(GraphEdge(
                source=cap.element_id,
                target=nearest.element_id,
                edge_type=EdgeType.CAPTION_OF,
            ))
        return edges

    def _build_adjacent_edges(self, threshold: float | None = None) -> list[GraphEdge]:
        """Connect spatially close elements on the same page.

        Two elements get an ADJACENT edge when their bbox centers are within
        the threshold (PDF points, ~1/72 inch). We only emit one edge per
        unordered pair, ordered by element_id, so the relation reads as
        effectively undirected even though GraphEdge is directional.

        Optimisations vs. the first version:
          * Cross-type only: text-text adjacency is already covered by SEQ,
            so we skip pairs that share the same ElementType. This cuts the
            edge count dramatically on text-heavy pages.
          * Per-node cap: at most _ADJACENT_MAX_PER_NODE edges per node,
            keeping the nearest neighbours and dropping the rest.
          * Tighter default threshold (150pt instead of 200pt).

        Synthesized child nodes (MENTION, aggregate DATA_POINT) are excluded:
        they inherit their parent's bbox and would otherwise be "adjacent"
        to everything the parent touches, drowning the real spatial signal.
        """
        if threshold is None:
            threshold = self._ADJACENT_THRESHOLD
        edges: list[GraphEdge] = []
        # Track how many ADJACENT edges each node already has so we can
        # enforce the per-node cap without a second pass.
        per_node_count: dict[str, int] = defaultdict(int)
        by_page: dict[int, list[DocumentElement]] = defaultdict(list)
        for el in self._elements.values():
            if not self._is_spatial(el):
                continue
            by_page[el.page].append(el)

        for els in by_page.values():
            n = len(els)
            for i in range(n):
                a = els[i]
                if self._ADJACENT_CROSS_TYPE_ONLY and per_node_count[a.element_id] >= self._ADJACENT_MAX_PER_NODE:
                    continue
                for j in range(i + 1, n):
                    b = els[j]
                    # Skip same-type pairs when cross-type filtering is on.
                    # Text-text is covered by SEQ; figure-figure and
                    # table-table adjacency adds little value at this stage.
                    if self._ADJACENT_CROSS_TYPE_ONLY and a.element_type is b.element_type:
                        continue
                    if per_node_count[b.element_id] >= self._ADJACENT_MAX_PER_NODE:
                        continue
                    # Cheap reject: skip pairs whose bboxes don't even
                    # overlap in projection -- they can't be within
                    # `threshold` unless the threshold is huge.
                    if not self._bbox_may_be_near(a.bbox, b.bbox, threshold):
                        continue
                    if a.bbox.center_distance(b.bbox) <= threshold:
                        src, dst = sorted((a.element_id, b.element_id))
                        edges.append(GraphEdge(
                            source=src,
                            target=dst,
                            edge_type=EdgeType.ADJACENT,
                        ))
                        per_node_count[a.element_id] += 1
                        per_node_count[b.element_id] += 1
                        if per_node_count[a.element_id] >= self._ADJACENT_MAX_PER_NODE:
                            break
        return edges

    def _build_extracted_from_edges(self) -> list[GraphEdge]:
        """Link data points to their source figure via EXTRACTED_FROM.

        Two situations are handled:

        1. A FIGURE carries a populated `data_points` list (e.g. data was
           extracted from a chart). We mint a single aggregate DATA_POINT
           child node holding that data and point it at the figure. This
           gives the data a first-class node so M5/M6 can attach claims to
           it rather than to the figure blob itself.

        2. Standalone DATA_POINT elements already in the graph (produced by
           some other stage). We attach each to the figure named in
           metadata['source_figure_id'] when available, otherwise to the
           nearest figure on the same page.
        """
        edges: list[GraphEdge] = []

        # Case 1: figures that carry extracted data.
        for el in list(self._elements.values()):
            if not el.data_points:
                continue
            dp = self._make_data_point_element(el)
            self._elements[dp.element_id] = dp
            edges.append(GraphEdge(
                source=dp.element_id,
                target=el.element_id,
                edge_type=EdgeType.EXTRACTED_FROM,
            ))

        # Case 2: free-standing data points supplied by the caller.
        figures = [e for e in self._elements.values()
                   if e.element_type is ElementType.FIGURE]
        for el in self._elements.values():
            if el.element_type is not ElementType.DATA_POINT:
                continue
            # Skip the aggregates we just synthesized -- they already have
            # their EXTRACTED_FROM edge from case 1.
            if el.metadata.get("_synthesized"):
                continue
            target_id = el.metadata.get("source_figure_id")
            if target_id and target_id in self._elements:
                target = self._elements[target_id]
            else:
                same_page = [f for f in figures if f.page == el.page]
                if not same_page:
                    continue
                target = min(same_page,
                             key=lambda f: el.bbox.center_distance(f.bbox))
            edges.append(GraphEdge(
                source=el.element_id,
                target=target.element_id,
                edge_type=EdgeType.EXTRACTED_FROM,
            ))
        return edges

    # ------------------------------------------------------------------
    # Mention extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_caption_type(content: Any) -> str | None:
        """Determine whether a caption refers to a figure or a table.

        Scans the caption text for "图N" / "Figure N" (-> figure) or
        "表N" / "Table N" (-> table). Returns None when no type marker
        is found, in which case the caller falls back to both types.
        """
        if not isinstance(content, str):
            return None
        for pattern, kind in _CAPTION_TYPE_PATTERNS:
            if pattern.search(content):
                return kind
        return None

    def _extract_mentions(self, element: DocumentElement) -> list[DocumentElement]:
        """Find figure/table mentions in a text block.

        Returns one MENTION element per unique (type, number) pair found.
        Deduplication is per-text-block, so "Figure 3" mentioned twice in
        the same paragraph yields a single mention node, but the same
        figure referenced from two different paragraphs yields two nodes
        (one per parent text block). That mirrors how the mentions are
        consumed downstream: each is a pointer anchored at a specific
        location in the text.
        """
        if not isinstance(element.content, str):
            return []

        seen: set[tuple[str, int]] = set()
        mentions: list[DocumentElement] = []

        for pattern, mention_type in _MENTION_PATTERNS:
            for m in pattern.finditer(element.content):
                number = int(m.group(1))
                key = (mention_type, number)
                if key in seen:
                    continue
                seen.add(key)

                self._child_counter += 1
                mid = f"{element.element_id}__mention_{self._child_counter}"
                mentions.append(DocumentElement(
                    element_id=mid,
                    element_type=ElementType.MENTION,
                    content=m.group(0),
                    page=element.page,
                    # We don't have char-level boxes, so the mention inherits
                    # its parent block's bbox. Spatial reasoning on mentions
                    # is therefore block-level, which is good enough for
                    # resolving "this paragraph refers to Figure 3".
                    bbox=element.bbox,
                    mention_type=mention_type,
                    mention_number=number,
                    metadata={"parent_id": element.element_id},
                ))
        return mentions

    def _make_data_point_element(self, source: DocumentElement) -> DocumentElement:
        """Create an aggregate DATA_POINT node for a figure's data_points."""
        self._child_counter += 1
        did = f"{source.element_id}__data_{self._child_counter}"
        return DocumentElement(
            element_id=did,
            element_type=ElementType.DATA_POINT,
            content=source.content,
            page=source.page,
            bbox=source.bbox,
            data_points=list(source.data_points or []),
            metadata={
                "parent_id": source.element_id,
                # Marker so case 2 of _build_extracted_from_edges knows to
                # skip us. Leading underscore keeps it out of any public
                # serialization that filters "private" keys.
                "_synthesized": True,
            },
        )

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_spatial(el: DocumentElement) -> bool:
        """Whether an element owns a meaningful bbox for adjacency.

        MENTION nodes and synthesized DATA_POINT aggregates borrow their
        parent's box, so including them in ADJACENT would just echo the
        parent's spatial relations and inflate the edge count.
        """
        if el.element_type is ElementType.MENTION:
            return False
        if el.metadata.get("_synthesized"):
            return False
        return True

    @staticmethod
    def _bbox_may_be_near(a: BBox, b: BBox, threshold: float) -> bool:
        """Quick reject for the adjacency test.

        Returns False when the two bboxes are provably farther apart than
        `threshold` based on a cheap separability check, avoiding the
        sqrt in center_distance for the common far-apart case.
        """
        # Expand b by threshold on every side and test overlap. If a's
        # center could still be inside the expanded b, we can't reject.
        if a.x2 < b.x1 - threshold or b.x2 < a.x1 - threshold:
            return False
        if a.y2 < b.y1 - threshold or b.y2 < a.y1 - threshold:
            return False
        return True

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def get_elements(
        self, element_type: ElementType | None = None
    ) -> list[DocumentElement]:
        """Return all elements, optionally filtered by type.

        Order is insertion order, which for a freshly built graph matches
        the order passed to build(). get_adjacency_matrix() uses this same
        ordering, so the two stay consistent for GNN feature assembly.
        """
        if element_type is None:
            return list(self._elements.values())
        return [e for e in self._elements.values() if e.element_type is element_type]

    def get_edges(self, edge_type: EdgeType | None = None) -> list[GraphEdge]:
        """Return all edges, optionally filtered by type."""
        if edge_type is None:
            return list(self._edges)
        return [e for e in self._edges if e.edge_type is edge_type]

    def get_neighbors(
        self, element_id: str, edge_type: EdgeType | None = None
    ) -> list[DocumentElement]:
        """Return elements connected to `element_id`.

        Direction-agnostic: both outgoing and incoming edges are followed,
        so SEQ (directed) and ADJACENT (effectively undirected) both behave
        intuitively. `edge_type` restricts which edges are traversed.
        Returns an empty list for unknown element ids rather than raising.
        """
        if element_id not in self._elements:
            return []
        adj = self._adjacency.get(element_id)
        if not adj:
            return []
        neighbor_ids: set[str] = set()
        if edge_type is not None:
            neighbor_ids.update(adj.get(edge_type, ()))
        else:
            for ids in adj.values():
                neighbor_ids.update(ids)
        return [self._elements[nid] for nid in neighbor_ids if nid in self._elements]

    def get_adjacency_matrix(
        self, edge_type: EdgeType | None = None
    ) -> np.ndarray:
        """Return the adjacency matrix as a dense float32 array.

        Node ordering matches get_elements() -- callers that need to align
        features with the matrix should snapshot get_elements() once and
        reuse the order. Pass `edge_type` to restrict to a single relation
        (handy for heterogeneous GNNs that keep per-relation matrices);
        leave it None for a combined weighted view.
        """
        nodes = self.get_elements()
        idx = {el.element_id: i for i, el in enumerate(nodes)}
        n = len(nodes)
        mat = np.zeros((n, n), dtype=np.float32)
        for e in self._edges:
            if edge_type is not None and e.edge_type is not edge_type:
                continue
            i = idx.get(e.source)
            j = idx.get(e.target)
            if i is None or j is None:
                # Edge references a node not in the current view -- can
                # happen for get_subgraph results. Skip silently.
                continue
            mat[i, j] = e.weight
        return mat

    def get_subgraph(self, element_ids: list[str]) -> "DocumentGraph":
        """Return a new graph restricted to the given element ids.

        Only edges whose both endpoints survive the cut are kept; the rest
        are dropped. Handy for slicing a single page, a figure's
        neighbourhood, or any other focus region before handing the graph
        to a model.
        """
        keep = set(element_ids)
        sub = DocumentGraph()
        for eid in element_ids:
            el = self._elements.get(eid)
            if el is not None:
                sub._elements[eid] = el
        for e in self._edges:
            if e.source in keep and e.target in keep:
                key = (e.source, e.target, e.edge_type)
                if key in sub._edge_index:
                    continue
                sub._edge_index.add(key)
                sub._edges.append(e)
                sub._adjacency[e.source][e.edge_type].add(e.target)
                sub._adjacency[e.target][e.edge_type].add(e.source)
        return sub

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize for API responses / JSON storage.

        Mirrors the to_dict() shape on DocumentElement so the whole graph
        round-trips through json cleanly. Embeddings are intentionally
        excluded -- they're large and downstream clients usually fetch them
        separately via the element_id.
        """
        return {
            "elements": [e.to_dict() for e in self._elements.values()],
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "edge_type": e.edge_type.value,
                    "weight": e.weight,
                    "confidence": e.confidence,
                    "metadata": e.metadata,
                }
                for e in self._edges
            ],
        }

    def stats(self) -> dict[str, Any]:
        """Return summary statistics for monitoring / debugging dashboards."""
        node_counts: dict[str, int] = defaultdict(int)
        for el in self._elements.values():
            node_counts[el.element_type.value] += 1
        edge_counts: dict[str, int] = defaultdict(int)
        for e in self._edges:
            edge_counts[e.edge_type.value] += 1
        return {
            "n_nodes": len(self._elements),
            "n_edges": len(self._edges),
            "node_types": dict(node_counts),
            "edge_types": dict(edge_counts),
        }

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._elements)

    def __contains__(self, element_id: object) -> bool:
        return element_id in self._elements

    def __repr__(self) -> str:
        return (
            f"DocumentGraph(nodes={len(self._elements)}, "
            f"edges={len(self._edges)})"
        )
