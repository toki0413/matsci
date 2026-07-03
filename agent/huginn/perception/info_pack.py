"""M6: Information Package Assembler.

Final stage of the document understanding pipeline. Walks the validated
graph (after M4 references and M5 cross-validation have run) and bundles
related elements into InformationPackage units -- the deliverable a
downstream LLM consumes as a single coherent context block.

Packaging strategy:
  * Seed each package at a CLAIM node and BFS over the cross-modal edges
    (CONTAINS, REFERENCES, EXTRACTED_FROM, SUPPORTS / CONTRADICTS /
    INCONCLUSIVE) to pull in the text, figures and data points that back
    the claim.
  * Elements are assigned exclusively: once a node is absorbed by a claim
    package it is not re-emitted in any other package.
  * Text blocks left over (no claim reached them) are grouped by reading
    order (SEQ) into plain-text packages so nothing falls on the floor.

The assembler is model-optional: pass an llm_client to get LLM-written
summaries, or leave it None for deterministic rule-based summaries. This
module only walks the graph -- no predictions, no I/O.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from huginn.perception.doc_types import (
    DocumentElement,
    GraphEdge,
    EdgeType,
    ElementType,
    InformationPackage,
)
from huginn.perception.document_graph import DocumentGraph


# Edges we're willing to walk during claim BFS. Deliberately excludes SEQ
# (reading order, not semantic), CAPTION_OF (already implied via REFERENCES
# once the caption's mention is resolved) and ADJACENT (purely spatial).
_BFS_EDGE_TYPES: frozenset[EdgeType] = frozenset({
    EdgeType.CONTAINS,
    EdgeType.REFERENCES,
    EdgeType.EXTRACTED_FROM,
    EdgeType.SUPPORTS,
    EdgeType.CONTRADICTS,
    EdgeType.INCONCLUSIVE,
})

# Validation edges carry the outcome of checking a claim against data.
_VALIDATION_EDGE_TYPES: frozenset[EdgeType] = frozenset({
    EdgeType.SUPPORTS,
    EdgeType.CONTRADICTS,
    EdgeType.INCONCLUSIVE,
})


class InfoPackAssembler:
    """Bundle a DocumentGraph into InformationPackage units.

    Usage::

        assembler = InfoPackAssembler()
        packages = assembler.assemble(graph)
        # each package is a self-contained context block
    """

    def __init__(self, llm_client: Any | None = None) -> None:
        # Optional: an object exposing .complete(prompt) -> str, or a plain
        # callable. When absent we fall back to rule-based summaries.
        self.llm = llm_client
        # Per-assemble caches. Populated at the top of assemble() so the
        # BFS helpers keep the public method signatures the caller expects.
        self._element_map: dict[str, DocumentElement] = {}
        self._adj: dict[str, set[str]] = {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def assemble(self, graph: DocumentGraph) -> list[InformationPackage]:
        """Assemble information packages from the document graph.

        Seeds: CLAIM elements -> cross-modal packages
        Remainder: orphan text blocks -> plain-text packages
        """
        # Index elements by id once -- cheaper than re-iterating the graph
        # for every BFS frontier expansion.
        self._element_map = {e.element_id: e for e in graph.get_elements()}
        self._adj = self._build_adjacency(graph)

        visited: set[str] = set()
        packages: list[InformationPackage] = []

        # Claims first. Sort for deterministic output across runs.
        claims = sorted(
            graph.get_elements(ElementType.CLAIM),
            key=lambda e: (e.page, e.element_id),
        )
        for claim in claims:
            if claim.element_id in visited:
                # Already swallowed by an earlier claim's BFS.
                continue
            pkg = self._bfs_package(claim.element_id, graph, visited)
            if pkg is not None:
                packages.append(pkg)

        # Whatever text wasn't claimed gets packed by reading order.
        packages.extend(self._collect_orphan_packages(graph, visited))

        # Drop caches so a stale graph doesn't leak into the next call.
        self._element_map = {}
        self._adj = {}
        return packages

    # ------------------------------------------------------------------
    # Claim-seeded BFS packaging
    # ------------------------------------------------------------------

    def _bfs_package(
        self,
        seed_id: str,
        graph: DocumentGraph,
        visited: set[str],
    ) -> InformationPackage | None:
        """BFS from a claim seed, collecting connected elements."""
        collected = self._bfs_collect(seed_id, visited)
        if not collected:
            return None

        pkg = self._assemble_package(f"pack_claim_{seed_id}", collected, graph)

        # Guarantee: every emitted package carries at least one text block.
        # A claim that BFS couldn't link back to any TEXT node still has its
        # own content, which is good enough as a textual anchor.
        if not pkg.text_blocks:
            seed_el = self._element_map.get(seed_id)
            if seed_el and isinstance(seed_el.content, str) and seed_el.content:
                pkg.text_blocks.append(seed_el.content)

        if not pkg.text_blocks:
            # Degenerate seed with no content -- release and skip rather
            # than emit an empty package.
            for eid in collected:
                visited.discard(eid)
            return None

        pkg.summary = self._build_summary(pkg)
        return pkg

    def _bfs_collect(self, seed_id: str, visited: set[str]) -> list[str]:
        """Plain BFS over the cached adjacency, marking nodes as visited."""
        queue: deque[str] = deque([seed_id])
        visited.add(seed_id)
        collected: list[str] = []
        while queue:
            cur = queue.popleft()
            collected.append(cur)
            for nxt in self._adj.get(cur, ()):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        return collected

    # ------------------------------------------------------------------
    # Orphan text packaging
    # ------------------------------------------------------------------

    def _collect_orphan_packages(
        self,
        graph: DocumentGraph,
        visited: set[str],
    ) -> list[InformationPackage]:
        """Group remaining text blocks by SEQ adjacency."""
        orphans = [
            e for e in graph.get_elements(ElementType.TEXT)
            if e.element_id not in visited
        ]
        if not orphans:
            return []

        # SEQ is a directed reading-order edge, but for grouping we want
        # connected components, so treat it as undirected here.
        seq_adj: dict[str, set[str]] = defaultdict(set)
        for edge in graph.get_edges(EdgeType.SEQ):
            seq_adj[edge.source].add(edge.target)
            seq_adj[edge.target].add(edge.source)

        orphan_ids = {e.element_id for e in orphans}
        local_visited: set[str] = set()
        packages: list[InformationPackage] = []
        idx = 0

        # Iterate in insertion order so numbering is stable.
        for el in orphans:
            if el.element_id in local_visited:
                continue
            # Pull the whole SEQ-connected component this block belongs to.
            comp: list[str] = []
            queue: deque[str] = deque([el.element_id])
            local_visited.add(el.element_id)
            while queue:
                cur = queue.popleft()
                comp.append(cur)
                for nxt in seq_adj.get(cur, ()):
                    if nxt in orphan_ids and nxt not in local_visited:
                        local_visited.add(nxt)
                        queue.append(nxt)
            idx += 1
            pkg = self._assemble_package(f"pack_{idx}", comp, graph)
            pkg.summary = self._build_summary(pkg)
            packages.append(pkg)

        return packages

    # ------------------------------------------------------------------
    # Package construction
    # ------------------------------------------------------------------

    def _assemble_package(
        self,
        package_id: str,
        element_ids: list[str],
        graph: DocumentGraph,
    ) -> InformationPackage:
        """Populate an InformationPackage from a set of element ids."""
        pkg = InformationPackage(package_id=package_id)
        pkg_ids = set(element_ids)

        elements = [
            self._element_map[eid] for eid in element_ids
            if eid in self._element_map
        ]
        pkg.elements = list(elements)

        # A figure's data_points are mirrored onto a synthesized DATA_POINT
        # child by the graph builder. If that child is in the package we
        # already count the data there, so skip the figure's own copy to
        # avoid double counting.
        figs_with_dp_child: set[str] = {
            el.metadata.get("parent_id")
            for el in elements
            if el.element_type is ElementType.DATA_POINT
            and el.metadata.get("parent_id")
        }

        for el in elements:
            if el.element_type is ElementType.TEXT:
                if isinstance(el.content, str) and el.content:
                    pkg.text_blocks.append(el.content)
            elif el.element_type is ElementType.FIGURE:
                if isinstance(el.content, str) and el.content:
                    pkg.figures.append(el.content)
                if el.data_points and el.element_id not in figs_with_dp_child:
                    pkg.data_points.extend(el.data_points)
            elif el.element_type is ElementType.DATA_POINT:
                if el.data_points:
                    pkg.data_points.extend(el.data_points)
            elif el.element_type is ElementType.CLAIM:
                if el.claim_data:
                    pkg.claims.append(el.claim_data)

        # Record validation verdicts for claims in this package. We anchor
        # on the claim (edge source) so a verdict is reported exactly once,
        # with the package that owns the claim.
        edges: list[GraphEdge] = graph.get_edges()
        for edge in edges:
            if edge.edge_type not in _VALIDATION_EDGE_TYPES:
                continue
            if edge.source not in pkg_ids:
                continue
            pkg.validation_results.append({
                "claim_id": edge.source,
                "target_id": edge.target,
                "relation": edge.edge_type.value,
                "confidence": edge.confidence,
                "metadata": edge.metadata,
            })

        return pkg

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    def _build_summary(self, package: InformationPackage) -> str:
        """Generate a short summary for the package."""
        if self.llm is not None:
            return self._llm_summary(package)
        return self._rule_summary(package)

    def _rule_summary(self, package: InformationPackage) -> str:
        """Fallback summary when no LLM is wired up."""
        if not package.claims:
            # Plain-text package -- describe the text we have.
            n = len(package.text_blocks)
            snippet = package.text_blocks[0][:80].rstrip() if package.text_blocks else ""
            base = f"Plain-text package of {n} text block(s)"
            if snippet:
                base += f": {snippet}..."
            return base

        parts: list[str] = []

        # Lead with the claim's conclusion if we can find one.
        lead = None
        for c in package.claims:
            lead = c.get("conclusion") or c.get("statement") or c.get("claim") or c.get("text")
            if lead:
                break
        if lead:
            parts.append(str(lead))

        if package.figures:
            parts.append(f"validated against {len(package.figures)} figure(s)")

        if package.validation_results:
            n_sup = sum(1 for v in package.validation_results
                        if v.get("relation") == EdgeType.SUPPORTS.value)
            n_con = sum(1 for v in package.validation_results
                        if v.get("relation") == EdgeType.CONTRADICTS.value)
            n_inc = sum(1 for v in package.validation_results
                        if v.get("relation") == EdgeType.INCONCLUSIVE.value)
            bits: list[str] = []
            if n_sup:
                bits.append(f"{n_sup} supported")
            if n_con:
                bits.append(f"{n_con} contradicted")
            if n_inc:
                bits.append(f"{n_inc} inconclusive")
            if bits:
                parts.append(", ".join(bits))

        if not parts:
            parts.append(
                f"{len(package.claims)} claim(s) with {len(package.figures)} figure(s)"
            )

        return "; ".join(parts)[:300]

    def _llm_summary(self, package: InformationPackage) -> str:
        """Ask the wired-up LLM for a one-line summary."""
        prompt = self._build_llm_prompt(package)
        try:
            if hasattr(self.llm, "complete"):
                text = self.llm.complete(prompt)
            else:
                text = self.llm(prompt)
            text = str(text).strip()
            if text:
                return text[:400]
        except Exception:
            # Swallow LLM errors and fall through to the rule-based path --
            # better a mediocre summary than a crashed pipeline.
            pass
        return self._rule_summary(package)

    def _build_llm_prompt(self, package: InformationPackage) -> str:
        """Compose a compact prompt describing the package for summarization."""
        claim_lines: list[str] = []
        for i, c in enumerate(package.claims[:5], 1):
            line = c.get("conclusion") or c.get("statement") or str(c)
            claim_lines.append(f"  {i}. {line}")

        text_snip = ""
        if package.text_blocks:
            text_snip = package.text_blocks[0][:200]
            if len(package.text_blocks[0]) > 200:
                text_snip += "..."

        return (
            "Summarize the following information package from a materials "
            "science paper in one concise sentence (max 30 words). Focus on "
            "the claim and how it was validated.\n\n"
            f"Claims:\n" + "\n".join(claim_lines) + "\n\n"
            f"Figures: {len(package.figures)}\n"
            f"Data points: {len(package.data_points)}\n"
            f"Validation results: {len(package.validation_results)}\n"
            f"Text snippet: {text_snip}\n\n"
            "Summary:"
        )

    # ------------------------------------------------------------------
    # Graph helpers
    # ------------------------------------------------------------------

    def _build_adjacency(self, graph: DocumentGraph) -> dict[str, set[str]]:
        """Undirected adjacency over BFS-eligible edges only.

        Built once per assemble() call and reused across all claim BFS runs
        so we don't rescan the edge list for every frontier expansion.
        """
        adj: dict[str, set[str]] = defaultdict(set)
        for edge in graph.get_edges():
            if edge.edge_type not in _BFS_EDGE_TYPES:
                continue
            adj[edge.source].add(edge.target)
            adj[edge.target].add(edge.source)
        return adj
