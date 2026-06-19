"""Tests for built-in knowledge-base seed documents."""

import tempfile
from pathlib import Path

from huginn.knowledge import KnowledgeBase, seed_knowledge_base


class TestKnowledgeSeed:
    def test_seed_adds_documents(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            kb = KnowledgeBase(Path(tmp) / "kb")
            result = seed_knowledge_base(kb)
            assert result["added"] > 0
            assert result["failed"] == 0

            docs = kb.list_documents()
            seed_docs = [d for d in docs if d["doc_id"].startswith("seed:")]
            assert len(seed_docs) == result["added"]

    def test_seed_is_idempotent(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            kb = KnowledgeBase(Path(tmp) / "kb")
            first = seed_knowledge_base(kb)
            assert first["added"] > 0

            second = seed_knowledge_base(kb)
            assert second["added"] == 0
            assert second["skipped"] == first["added"]
            assert second["failed"] == 0

    def test_seed_force_reloads(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            kb = KnowledgeBase(Path(tmp) / "kb")
            first = seed_knowledge_base(kb)
            assert first["added"] > 0

            forced = seed_knowledge_base(kb, force=True)
            assert forced["added"] == first["added"]
            assert forced["skipped"] == 0
            assert forced["failed"] == 0

    def test_seed_query_returns_relevant_chunks(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            kb = KnowledgeBase(Path(tmp) / "kb")
            seed_knowledge_base(kb)

            # ENCUT is split into a later chunk; retrieve enough results.
            results = kb.query("VASP ENCUT", top_k=20)
            assert len(results) > 0
            assert any("ENCUT" in r["text"] for r in results)

    def test_seed_geometry_validation_query(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            kb = KnowledgeBase(Path(tmp) / "kb")
            seed_knowledge_base(kb)

            results = kb.query("RMSD equivalent structures", top_k=5)
            assert len(results) > 0
            assert any("RMSD" in r["text"] for r in results)

    def test_query_cache_returns_same_result(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            kb = KnowledgeBase(Path(tmp) / "kb")
            seed_knowledge_base(kb)

            first = kb.query("VASP ENCUT", top_k=5)
            second = kb.query("VASP ENCUT", top_k=5)
            assert [r["chunk_id"] for r in first] == [r["chunk_id"] for r in second]
            assert len(kb._query_cache) == 1
