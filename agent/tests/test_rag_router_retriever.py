"""Unit tests for huginn/rag/router_retriever.py."""

from __future__ import annotations

import json

import pytest

from huginn.rag.router_retriever import HierarchicalRetriever


class _FakeVectorStore:
    """In-memory vector store for testing HierarchicalRetriever."""

    def __init__(self, targeted=None, general=None):
        self.targeted = targeted or []
        self.general = general or []

    def search(self, query, top_k=5, filter_dict=None):
        if filter_dict:
            return [
                r
                for r in self.targeted
                if all(r.get(k) == v for k, v in filter_dict.items())
            ][:top_k]
        return self.general[:top_k]


class TestRouteDetection:
    def test_detect_vasp(self):
        store = _FakeVectorStore()
        ret = HierarchicalRetriever(store)
        sw, method = ret._detect_route("How to set VASP INCAR tags?")
        assert sw == "VASP"
        assert method is None

    def test_detect_method(self):
        store = _FakeVectorStore()
        ret = HierarchicalRetriever(store)
        sw, method = ret._detect_route("RESP charge fitting procedure")
        assert method == "RESP"

    def test_no_route(self):
        store = _FakeVectorStore()
        ret = HierarchicalRetriever(store)
        sw, method = ret._detect_route("hello world")
        assert sw is None and method is None


class TestSearch:
    def test_search_routes_and_deduplicates(self):
        targeted = [
            {"id": "vasp-1", "text": "ENCUT guide", "software_tags": "VASP"},
            {"id": "vasp-2", "text": "ALGO guide", "software_tags": "VASP"},
        ]
        general = [
            {"id": "vasp-1", "text": "ENCUT guide", "software_tags": "VASP"},
            {"id": "gen-1", "text": "general dft"},
        ]
        store = _FakeVectorStore(targeted=targeted, general=general)
        ret = HierarchicalRetriever(store)
        result = ret.search("VASP ENCUT", top_k=5)

        assert result["route"]["software"] == "VASP"
        assert result["targeted_count"] == 2
        assert result["general_count"] == 2
        ids = {r.get("id") for r in result["results"]}
        assert ids == {"vasp-1", "vasp-2", "gen-1"}
        assert "VASP" in result["routing_reason"]

    def test_search_no_route(self):
        store = _FakeVectorStore(general=[{"id": "g1", "text": "general"}])
        ret = HierarchicalRetriever(store)
        result = ret.search("something abstract")
        assert result["route"] == {"software": None, "method": None}
        assert "general search" in result["routing_reason"]

    def test_search_store_exception(self):
        class _BrokenStore:
            def search(self, **kwargs):
                raise RuntimeError("boom")

        ret = HierarchicalRetriever(_BrokenStore())
        result = ret.search("VASP", top_k=2)
        assert result["targeted_count"] == 0
        assert result["general_count"] == 0
        assert result["results"] == []


class TestIndexLoading:
    def test_load_index_from_path(self, tmp_path):
        index = {
            "software": {"VASP": ["incar", "poscar"]},
            "method": {"DFT": ["pbe"]},
        }
        path = tmp_path / "index.json"
        path.write_text(json.dumps(index), encoding="utf-8")

        ret = HierarchicalRetriever(_FakeVectorStore(), index_path=str(path))
        assert ret._index == index
