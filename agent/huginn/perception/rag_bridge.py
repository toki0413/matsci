"""RAG Bridge: feed DocumentGraph info packages into the knowledge base.

After the DocGraph pipeline produces InformationPackage units, this module
converts each package into a searchable document and ingests it into the
KnowledgeBase (or VectorStore) so the agent can retrieve relevant context
during conversations.

Each info package becomes one KB document with:
  - text: the package summary + all text blocks concatenated
  - metadata: source PDF, page range, claim metrics, validation verdicts
  - tags: figure/table presence, claim count, support/contradict counts

The bridge is KB-optional: if no knowledge base is wired up, ``ingest``
is a no-op that returns 0. This keeps the pipeline runnable in test
setups without a running ChromaDB instance.
"""

from __future__ import annotations

import logging
from typing import Any

from huginn.perception.doc_types import InformationPackage
from huginn.perception.document_graph import DocumentGraph

logger = logging.getLogger(__name__)


class RAGBridge:
    """Convert info packages to KB documents and ingest them.

    Usage::

        bridge = RAGBridge(kb=my_knowledge_base)
        n = bridge.ingest(packages, document_id="doc_abc")
    """

    def __init__(self, kb: Any | None = None) -> None:
        # The KB can be a KnowledgeBase, a VectorStore, or anything that
        # exposes add_document(text, metadata). When None, ingest is a no-op.
        self.kb = kb

    def ingest(
        self,
        packages: list[InformationPackage],
        document_id: str = "",
        filename: str = "",
    ) -> int:
        """Push all packages into the knowledge base.

        Returns the number of documents actually ingested. Returns 0
        immediately when no KB is configured.
        """
        if self.kb is None or not packages:
            return 0

        count = 0
        for pkg in packages:
            text = self._package_to_text(pkg)
            if not text.strip():
                continue
            metadata = self._package_metadata(pkg, document_id, filename)
            try:
                self._add_document(text, metadata)
                count += 1
            except Exception as exc:
                logger.debug("KB ingest failed for %s: %s", pkg.package_id, exc)
        logger.info("RAGBridge: ingested %d/%d packages", count, len(packages))
        return count

    def _package_to_text(self, pkg: InformationPackage) -> str:
        """Flatten a package into a single text blob for embedding.

        Layout: summary first (most important for retrieval), then the
        text blocks, then a compact claim digest. Figures and data points
        are referenced by count — we don't embed raw image bytes or large
        numeric arrays.
        """
        parts: list[str] = []

        if pkg.summary:
            parts.append(pkg.summary)

        for block in pkg.text_blocks:
            parts.append(block)

        for claim in pkg.claims:
            metric = claim.get("metric")
            value = claim.get("value")
            unit = claim.get("unit")
            qualifier = claim.get("qualifier")
            conclusion = claim.get("conclusion", "")
            bits: list[str] = []
            if metric:
                bits.append(f"metric={metric}")
            if value is not None:
                bits.append(f"value={value}")
            if unit:
                bits.append(f"unit={unit}")
            if qualifier:
                bits.append(f"qualifier={qualifier}")
            if conclusion:
                bits.append(f"conclusion={conclusion[:120]}")
            if bits:
                parts.append("[CLAIM] " + ", ".join(bits))

        for vr in pkg.validation_results:
            verdict = vr.get("relation", "unknown")
            evidence = vr.get("metadata", {}).get("evidence", "")
            parts.append(f"[{verdict.upper()}] {evidence[:100]}")

        return "\n".join(parts)

    def _package_metadata(
        self, pkg: InformationPackage, document_id: str, filename: str
    ) -> dict[str, Any]:
        """Build KB metadata for a package."""
        n_supported = sum(
            1 for v in pkg.validation_results
            if v.get("relation") == "supports"
        )
        n_contradicted = sum(
            1 for v in pkg.validation_results
            if v.get("relation") == "contradicts"
        )
        n_inconclusive = sum(
            1 for v in pkg.validation_results
            if v.get("relation") == "inconclusive"
        )

        # Collect the set of pages this package touches.
        pages = sorted({el.page for el in pkg.elements if el.page is not None})

        return {
            "source": "docgraph",
            "document_id": document_id,
            "filename": filename,
            "package_id": pkg.package_id,
            "n_text_blocks": len(pkg.text_blocks),
            "n_figures": len(pkg.figures),
            "n_data_points": len(pkg.data_points),
            "n_claims": len(pkg.claims),
            "n_supported": n_supported,
            "n_contradicted": n_contradicted,
            "n_inconclusive": n_inconclusive,
            "pages": pages,
        }

    def _add_document(self, text: str, metadata: dict[str, Any]) -> None:
        """Dispatch to whichever KB interface we were given."""
        kb = self.kb
        # KnowledgeBase.add_document(text, metadata)
        add = getattr(kb, "add_document", None)
        if callable(add):
            add(text, metadata)
            return
        # VectorStore.ingest(texts, metadatas) -- batch interface
        ingest = getattr(kb, "ingest", None)
        if callable(ingest):
            ingest([text], [metadata])
            return
        # Last resort: treat as a callable
        if callable(kb):
            kb(text, metadata)
            return
        raise TypeError(
            f"KB object {type(kb).__name__} has no add_document / ingest method"
        )

    def query(
        self, query_text: str, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Search the KB for packages matching a query.

        Returns a list of ``{text, metadata, score}`` dicts. Returns
        an empty list when no KB is configured.
        """
        if self.kb is None:
            return []
        try:
            search = getattr(self.kb, "query", None) or getattr(self.kb, "search", None)
            if callable(search):
                results = search(query_text, top_k=top_k)
                if isinstance(results, list):
                    return results
                if isinstance(results, dict) and "results" in results:
                    return results["results"]
            return []
        except Exception as exc:
            logger.debug("KB query failed: %s", exc)
            return []
