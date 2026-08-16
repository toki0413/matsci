"""P2#2: BM25 按 domain 分片 — 有 domain 过滤时也能走关键词混合检索.

验证 _BM25Index.add 记录 domain、search 按 domain 分片过滤, 保证:
- 无 domain 时全量检索 (含所有域).
- 有 domain 时只对同域 chunk 计分, 不泄漏他域命中.
- 不存在 domain 时返回空.
"""

from __future__ import annotations

from huginn.knowledge.store import _BM25Index


def _build_index() -> _BM25Index:
    idx = _BM25Index()
    idx.add("a1", "高熵合金 doping 掺杂 提高强度", "material")
    idx.add("a2", "退火 annealing temperature 温度 影响相变", "material")
    idx.add("b1", "神经网络 neural network 训练 拟合", "ml")
    idx.add("b2", "卷积 cnn 图像分类", "ml")
    idx._build()
    return idx


def test_add_records_domain():
    idx = _build_index()
    assert idx._doc_domains == ["material", "material", "ml", "ml"]


def test_no_domain_searches_all():
    idx = _build_index()
    hits = idx.search("temperature 温度", top_k=10)
    assert any(cid == "a2" for cid, _ in hits)


def test_domain_filter_restricts_to_shard():
    idx = _build_index()
    hits = idx.search("temperature 温度", top_k=10, domain="material")
    assert hits
    # 只允许同域 chunk 出现
    assert all(cid.startswith("a") for cid, _ in hits)


def test_other_domain_not_leaked():
    idx = _build_index()
    # ml 域搜索"训练/拟合", 不应返回 material 域的 a1
    hits = idx.search("训练 fit 拟合", top_k=10, domain="ml")
    assert hits
    assert all(cid.startswith("b") for cid, _ in hits)


def test_nonexistent_domain_returns_empty():
    idx = _build_index()
    assert idx.search("temperature", top_k=10, domain="nonexistent") == []


def test_default_domain_param_backward_compatible():
    # 不传 domain (=None) 时行为与改造前一致
    idx = _build_index()
    hits = idx.search("cnn 卷积", top_k=10)
    assert any(cid == "b2" for cid, _ in hits)
