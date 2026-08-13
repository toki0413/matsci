"""VectorStore.search() 集成测试 — 覆盖 ChromaDB 原生 query + Rust fallback + 关键字回退.

不依赖真实 chromadb: 通过 monkeypatch 实例方法注入 fake collection / embedding,
覆盖 search() 的四条路径:
  1. 原生 query 成功 → 直接返回
  2. 原生 query 失败(HNSW 未建) → 降级到 Rust top-k 全量 fallback
  3. Rust fallback + metadata filter → 先过滤再打分
  4. 无 embedding 模型 → 关键字搜索回退
整合 `_rust_top_k` 桥接 (test_vector_store_rust_ext.py 已单测).
"""

from __future__ import annotations

import sys
import types

import pytest

from huginn.rag.vector_store import VectorStore


def _make_store(tmp_path):
    return VectorStore(persist_dir=str(tmp_path / "rag"))


def _install_fake_huginn_ext(monkeypatch, top_k=None, raise_import=False):
    if raise_import:
        monkeypatch.setitem(sys.modules, "huginn_ext", None)
        return
    ext = types.ModuleType("huginn_ext")
    if top_k is not None:
        ext.top_k = top_k
    monkeypatch.setitem(sys.modules, "huginn_ext", ext)


class _FullCol:
    """全量数据集合: -query 抛异常触发 fallback, -get 返回全部."""
    def __init__(self, ids, docs, metas, embs):
        self.ids, self.docs, self.metas, self.embs = ids, docs, metas, embs
        self.query_calls = 0

    def query(self, **kw):
        self.query_calls += 1
        raise RuntimeError("HNSW index not built")

    def get(self, include=None, limit=None):
        return {
            "ids": self.ids,
            "documents": self.docs,
            "metadatas": self.metas,
            "embeddings": self.embs,
        }


# ── 原生 query 成功 ──────────────────────────────────────────────────────

def test_search_native_query_success(tmp_path):
    store = _make_store(tmp_path)
    store._compute_embeddings = lambda texts: [[0.1, 0.2]]

    class Fake:
        def query(self, **kw):
            return {
                "ids": [["a", "b"]],
                "documents": [["doc a", "doc b"]],
                "metadatas": [[{"src": "a"}, {"src": "b"}]],
                "distances": [[0.1, 0.3]],
            }

        def get(self, **kw):
            raise AssertionError("原生 query 成功, 不应走到 get")

    store._get_collection = lambda: Fake()
    out = store.search("query text", top_k=2)
    assert len(out) == 2
    assert out[0] == {"id": "a", "document": "doc a", "metadata": {"src": "a"}, "distance": 0.1}
    assert out[1]["id"] == "b"


def test_search_native_query_truncates_to_top_k(tmp_path):
    store = _make_store(tmp_path)
    store._compute_embeddings = lambda texts: [[0.1, 0.2]]

    class Fake:
        def query(self, **kw):
            return {
                "ids": [["a", "b", "c"]],
                "documents": [["1", "2", "3"]],
                "metadatas": [[{}, {}, {}]],
                "distances": [[0.1, 0.2, 0.3]],
            }

    store._get_collection = lambda: Fake()
    out = store.search("q", top_k=2)
    assert [o["id"] for o in out] == ["a", "b"]


# ── 原生 query 失败 → Rust top-k fallback ────────────────────────────────

def test_search_falls_back_to_rust_top_k(monkeypatch, tmp_path):
    store = _make_store(tmp_path)
    store._compute_embeddings = lambda texts: [[0.1, 0.2]]

    def top_k(q, e, k):
        return [(0, 0.9), (1, 0.5)]

    _install_fake_huginn_ext(monkeypatch, top_k=top_k)
    col = _FullCol(
        ids=["a", "b"],
        docs=["doc a", "doc b"],
        metas=[{"x": 1}, {"x": 2}],
        embs=[[0.1, 0.2], [0.3, 0.4]],
    )
    store._get_collection = lambda: col
    out = store.search("q", top_k=2)
    assert col.query_calls == 1  # 原生 query 被尝试过
    assert len(out) == 2
    assert out[0]["id"] == "a"
    assert out[0]["distance"] == pytest.approx(0.1)  # 1.0 - 0.9
    assert out[1]["distance"] == pytest.approx(0.5)


def test_search_rust_fallback_respects_filter(monkeypatch, tmp_path):
    store = _make_store(tmp_path)
    store._compute_embeddings = lambda texts: [[0.1, 0.2]]

    def top_k(q, e, k):
        return [(0, 0.8)]  # 过滤后只剩 index 0

    _install_fake_huginn_ext(monkeypatch, top_k=top_k)
    col = _FullCol(
        ids=["a", "b"],
        docs=["doc a", "doc b"],
        metas=[{"cat": "metal"}, {"cat": "polymer"}],
        embs=[[0.1, 0.2], [0.3, 0.4]],
    )
    store._get_collection = lambda: col
    out = store.search("q", top_k=5, filter_dict={"cat": "polymer"})
    # 过滤后只剩 b(polymer), 重排 index 0 → b
    assert len(out) == 1
    assert out[0]["id"] == "b"
    assert out[0]["metadata"] == {"cat": "polymer"}


def test_search_rust_fallback_no_results_when_stack_empty(monkeypatch, tmp_path):
    """_rust_top_k 返回 None (ext 不可用) → search 返回 []."""
    store = _make_store(tmp_path)
    store._compute_embeddings = lambda texts: [[0.1, 0.2]]
    _install_fake_huginn_ext(monkeypatch, raise_import=True)
    col = _FullCol(
        ids=["a"], docs=["doc a"], metas=[{}], embs=[[0.1, 0.2]]
    )
    store._get_collection = lambda: col
    assert store.search("q") == []


# ── 无 embedding 模型 → 关键字搜索回退 ───────────────────────────────────

def test_search_keyword_no_embedding(tmp_path):
    store = _make_store(tmp_path)
    store._compute_embeddings = lambda texts: None

    class Fake:
        def get(self, include=None, limit=None):
            return {
                "ids": ["a", "b"],
                "documents": ["rust diffusion", "polymer chain"],
                "metadatas": [{"src": "a"}, {"src": "b"}],
            }

    store._get_collection = lambda: Fake()
    out = store.search("rust", top_k=2)
    assert len(out) == 2
    assert out[0]["document"] == "rust diffusion"
    assert out[0]["id"] == "a"


def test_search_keyword_empty_collection(tmp_path):
    store = _make_store(tmp_path)
    store._compute_embeddings = lambda texts: None

    class Fake:
        def get(self, include=None, limit=None):
            return {"ids": [], "documents": [], "metadatas": []}

    store._get_collection = lambda: Fake()
    assert store.search("anything") == []