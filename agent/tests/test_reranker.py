"""Tests for huginn.rag.reranker — two-stage retrieval with cross-encoder reranking.

All cross-encoder tests mock sentence-transformers via sys.modules so they
never trigger a real model download.
"""

from __future__ import annotations

import asyncio
import sys
import types
from unittest.mock import patch

import pytest

from huginn.rag.reranker import (
    CrossEncoderReranker,
    NoopReranker,
    RerankResult,
    get_reranker,
    reset_reranker,
)


@pytest.fixture(autouse=True)
def _reset_global_reranker():
    """每个用例前后清掉单例, 避免用例之间互相污染。"""
    reset_reranker()
    yield
    reset_reranker()


# ---------------------------------------------------------------------------
# RerankResult dataclass
# ---------------------------------------------------------------------------


def test_rerank_result_defaults():
    """RerankResult 只有 document/score 是必填, 其余有默认值。"""
    r = RerankResult(document="hello", score=0.5)
    assert r.document == "hello"
    assert r.score == 0.5
    assert r.metadata is None
    assert r.original_rank == 0


def test_rerank_result_full_fields():
    r = RerankResult(
        document="doc",
        score=1.2,
        metadata={"source": "vasp"},
        original_rank=3,
    )
    assert r.metadata == {"source": "vasp"}
    assert r.original_rank == 3


# ---------------------------------------------------------------------------
# NoopReranker
# ---------------------------------------------------------------------------


def test_noop_reranker_is_available():
    assert NoopReranker().is_available() is True


def test_noop_reranker_truncates_to_top_k():
    docs = ["a", "b", "c", "d", "e"]
    results = asyncio.run(NoopReranker().rerank("q", docs, top_k=3))
    assert len(results) == 3
    # 保持原顺序, original_rank 跟输入下标一致
    assert [r.document for r in results] == ["a", "b", "c"]
    assert [r.original_rank for r in results] == [0, 1, 2]
    # noop 不打分, 全部 0.0
    assert all(r.score == 0.0 for r in results)


def test_noop_reranker_top_k_larger_than_docs():
    docs = ["only"]
    results = asyncio.run(NoopReranker().rerank("q", docs, top_k=5))
    assert len(results) == 1
    assert results[0].document == "only"


def test_noop_reranker_empty_documents():
    results = asyncio.run(NoopReranker().rerank("q", [], top_k=3))
    assert results == []


# ---------------------------------------------------------------------------
# CrossEncoderReranker — mocked model
# ---------------------------------------------------------------------------


class _FakeCrossEncoder:
    """假 CrossEncoder: predict 按 doc 长度打分, 越长分越高。"""

    instances = []

    def __init__(self, model_name):
        self.model_name = model_name
        self.predict_calls = 0
        _FakeCrossEncoder.instances.append(self)

    def predict(self, pairs):
        self.predict_calls += 1
        # pairs: list[(query, doc)]
        return [float(len(doc)) for _, doc in pairs]


def _fake_sentence_transformers_module():
    mod = types.ModuleType("sentence_transformers")
    mod.CrossEncoder = _FakeCrossEncoder
    return mod


def _make_unavailable_module():
    """返回一个让 `from sentence_transformers import CrossEncoder` 抛 ImportError 的占位。

    sys.modules[name] = None 时, import name 会直接抛 ImportError。
    """
    return None


def test_cross_encoder_is_available_false_when_not_installed():
    _FakeCrossEncoder.instances.clear()
    reranker = CrossEncoderReranker()
    with patch.dict(sys.modules, {"sentence_transformers": _make_unavailable_module()}):
        assert reranker.is_available() is False
    assert reranker._model is None
    # 没装库时不应该实例化任何 CrossEncoder
    assert _FakeCrossEncoder.instances == []


def test_cross_encoder_is_available_true_with_mock_model():
    _FakeCrossEncoder.instances.clear()
    reranker = CrossEncoderReranker(model_name="fake/reranker")
    with patch.dict(sys.modules, {"sentence_transformers": _fake_sentence_transformers_module()}):
        assert reranker.is_available() is True
        assert reranker._model is not None
        assert reranker._model.model_name == "fake/reranker"


def test_cross_encoder_rerank_orders_by_score_descending():
    _FakeCrossEncoder.instances.clear()
    reranker = CrossEncoderReranker()
    docs = ["short", "medium length doc", "the longest document here"]
    with patch.dict(sys.modules, {"sentence_transformers": _fake_sentence_transformers_module()}):
        results = asyncio.run(reranker.rerank("query", docs, top_k=3))

    # predict 返回 doc 长度, 降序后: 最长 -> 最短
    assert [r.document for r in results] == [
        "the longest document here",
        "medium length doc",
        "short",
    ]
    # original_rank 指向输入里的原始下标
    assert [r.original_rank for r in results] == [2, 1, 0]
    # 分数严格递减
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_cross_encoder_rerank_truncates_to_top_k():
    _FakeCrossEncoder.instances.clear()
    reranker = CrossEncoderReranker()
    docs = ["a", "bb", "ccc", "dddd", "eeeee"]
    with patch.dict(sys.modules, {"sentence_transformers": _fake_sentence_transformers_module()}):
        results = asyncio.run(reranker.rerank("q", docs, top_k=2))

    assert len(results) == 2
    # 最长的两条
    assert [r.document for r in results] == ["eeeee", "dddd"]


def test_cross_encoder_rerank_empty_documents():
    reranker = CrossEncoderReranker()
    with patch.dict(sys.modules, {"sentence_transformers": _fake_sentence_transformers_module()}):
        results = asyncio.run(reranker.rerank("q", [], top_k=3))
    assert results == []


def test_cross_encoder_rerank_fallback_when_model_unavailable():
    """模型加载失败时, rerank 退回原顺序截断, 分数全 0。"""
    reranker = CrossEncoderReranker()
    docs = ["first", "second", "third"]
    with patch.dict(sys.modules, {"sentence_transformers": _make_unavailable_module()}):
        results = asyncio.run(reranker.rerank("q", docs, top_k=2))

    assert len(results) == 2
    assert [r.document for r in results] == ["first", "second"]
    assert [r.original_rank for r in results] == [0, 1]
    assert all(r.score == 0.0 for r in results)


def test_cross_encoder_rerank_passes_query_doc_pairs():
    """predict 拿到的应该是 (query, doc) 配对。"""
    _FakeCrossEncoder.instances.clear()
    reranker = CrossEncoderReranker()
    docs = ["x", "yy"]
    captured_pairs = []

    class _CapturingEncoder(_FakeCrossEncoder):
        def predict(self, pairs):
            captured_pairs.extend(pairs)
            return super().predict(pairs)

    mod = types.ModuleType("sentence_transformers")
    mod.CrossEncoder = _CapturingEncoder
    with patch.dict(sys.modules, {"sentence_transformers": mod}):
        asyncio.run(reranker.rerank("my query", docs, top_k=2))

    assert captured_pairs == [("my query", "x"), ("my query", "yy")]


# ---------------------------------------------------------------------------
# get_reranker / reset_reranker singleton
# ---------------------------------------------------------------------------


def test_get_reranker_default_returns_noop():
    r = get_reranker()
    assert isinstance(r, NoopReranker)


def test_get_reranker_cross_encoder_when_available():
    _FakeCrossEncoder.instances.clear()
    with patch.dict(sys.modules, {"sentence_transformers": _fake_sentence_transformers_module()}):
        r = get_reranker(use_cross_encoder=True)
    assert isinstance(r, CrossEncoderReranker)


def test_get_reranker_falls_back_to_noop_when_unavailable():
    with patch.dict(sys.modules, {"sentence_transformers": _make_unavailable_module()}):
        r = get_reranker(use_cross_encoder=True)
    assert isinstance(r, NoopReranker)


def test_get_reranker_is_singleton():
    r1 = get_reranker()
    r2 = get_reranker()
    assert r1 is r2


def test_reset_reranker_clears_singleton():
    r1 = get_reranker()
    reset_reranker()
    r2 = get_reranker()
    assert r1 is not r2


def test_reset_reranker_idempotent():
    """重复 reset 不报错, 之后仍能拿到新实例。"""
    reset_reranker()
    reset_reranker()
    assert get_reranker() is not None
