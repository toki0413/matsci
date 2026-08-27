"""WeKnora GraphRAG 炼化自检: KG 关系扩展接入 KB 检索.

最小可运行检查 (ponytail):
1. _get_kg 无 KG 文件返回 None 且缓存 (不重复探测)
2. _get_kg 有 project_kg.json (workspace 根) 时加载
3. _kg_relation_activate 用实体词补 chunks, 只留实体命中的
4. _kg_relation_activate 无 KG 时返回空列表不抛异常
"""
from __future__ import annotations

import tempfile
from pathlib import Path


def _make_kb_stub(root: Path, collection, model):
    """绕过 chromadb 构造 KB 实例, 只塞 GraphRAG 扩展需要的字段."""
    from huginn.knowledge.store import KnowledgeBase

    kb = object.__new__(KnowledgeBase)
    kb.root = root
    kb._kg = None
    kb.collection = collection
    kb._model = model
    kb._query_cache = None
    kb._semantic_cache = None
    kb._feedback_tracker = None
    return kb


class _StubModel:
    def encode(self, texts):
        # 对齐真实 _EmbeddingModel: 返回 numpy array (实现里会 .tolist())
        import numpy as np
        return np.array([[0.1] * 16 for _ in texts])


class _StubCollection:
    """query 返回固定候选: doc1 含 GaN, doc2 含 DFT, doc3 无关."""

    def __init__(self):
        self._docs = {
            "doc1": "GaN 的 Mg 掺杂空穴浓度研究",
            "doc2": "DFT 计算 GaN 能带结构",
            "doc3": "完全不相关的文本内容",
        }
        self._metas = {k: {"domain_tags": "[]"} for k in self._docs}

    def query(self, query_embeddings, n_results, include):
        ids = list(self._docs.keys())
        return {
            "ids": [ids],
            "documents": [[self._docs[i] for i in ids]],
            "metadatas": [[self._metas[i] for i in ids]],
            "distances": [[0.1] * len(ids)],
        }

    def count(self):
        return len(self._docs)


def _make_kg(root: Path) -> None:
    """建一个带 GaN/Mg 实体和关系的项目知识图谱."""
    from huginn.kg.graph import ProjectKnowledgeGraph

    kg = ProjectKnowledgeGraph(root)
    kg.add_entity("GaN", "material", source="test")
    kg.add_entity("Mg", "element", source="test")
    kg.add_entity("ZnO", "material", source="test")
    kg.add_relation("GaN", "Mg", "doped_with")
    kg.save()


def _test_get_kg_missing():
    """无 project_kg.json → None, 且缓存 False 不重复探测."""
    with tempfile.TemporaryDirectory() as tmp:
        kb = _make_kb_stub(Path(tmp), _StubCollection(), _StubModel())
        assert kb._get_kg() is None, "无 KG 文件应返回 None"
        assert kb._kg is False, "探测结果应缓存为 False"
        # 第二次调用仍 None, 且不抛
        assert kb._get_kg() is None


def _test_get_kg_found():
    """workspace 根有 project_kg.json → 加载成功."""
    with tempfile.TemporaryDirectory() as tmp:
        _make_kg(Path(tmp))
        kb = _make_kb_stub(Path(tmp), _StubCollection(), _StubModel())
        kg = kb._get_kg()
        assert kg is not None, "有 KG 文件应加载"
        assert kg.has_entity("GaN", "material"), "KG 应含 GaN 实体"


def _test_kg_relation_activate():
    """实体词命中补 chunks, 只留含实体的."""
    with tempfile.TemporaryDirectory() as tmp:
        _make_kg(Path(tmp))
        kb = _make_kb_stub(Path(tmp), _StubCollection(), _StubModel())
        # 首轮已命中 doc1 (chunks 里已有), 让扩展只补 doc2
        chunks = [
            {"chunk_id": "doc1", "text": "GaN 的 Mg 掺杂空穴浓度研究",
             "metadata": {"domain_tags": "[]"}, "distance": 0.1}
        ]
        activated = kb._kg_relation_activate(chunks, "GaN Mg doping", top_k=5)
        assert activated, "应返回 KG 扩展补的 chunks"
        for c in activated:
            assert c["chunk_id"] != "doc1", "扩展不应重复已命中的块"
            assert "GaN" in c["text"] or "DFT" in c["text"], "补充块应含图谱实体词"


def _test_kg_relation_activate_no_kg():
    """无 KG 时返回空列表, 不抛异常."""
    with tempfile.TemporaryDirectory() as tmp:
        kb = _make_kb_stub(Path(tmp), _StubCollection(), _StubModel())
        chunks = [{"chunk_id": "doc1", "text": "x", "metadata": {}, "distance": 0.1}]
        assert kb._kg_relation_activate(chunks, "query", 5) == []


if __name__ == "__main__":
    _test_get_kg_missing()
    print("[1/4] 无 KG 缓存 None OK")
    _test_get_kg_found()
    print("[2/4] 有 KG 加载 OK")
    _test_kg_relation_activate()
    print("[3/4] 实体扩展补 chunks OK")
    _test_kg_relation_activate_no_kg()
    print("[4/4] 无 KG 不抛异常 OK")
    print("\nAll GraphRAG distill self-checks passed.")
