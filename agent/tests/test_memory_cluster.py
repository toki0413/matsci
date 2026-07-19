"""P5 Memory Semantic Cluster & Compress 集成测试.

覆盖 5 个场景:
1. env=0 (HUGINN_MEMORY_CLUSTER=0) 时 maintenance 跳过 cluster step
2. archived=0 默认行为: 所有 alive memory 都可 retrieve (不破坏现有行为)
3. archived=1 时 retrieve 自动过滤, 不返回归档条目
4. cluster_memories + compress_clusters 端到端: 同主题聚类 → LLM 浓缩 → 原条目归档
5. env=1 + maintenance(cluster=True, llm_chat_fn) 真实闭环
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from huginn.memory.cluster import cluster_memories, compress_clusters
from huginn.memory.longterm import LongTermMemory


class _MockVectorStore:
    """覆盖 LongTermMemory 调用的 vector_store 接口.

    _compute_embeddings 返回 deterministic vectors: 同主题同向, 异主题异向.
    ingest/search 走 no-op, retrieve 走 FTS5 fallback, 不依赖真实 embedding 模型.
    """

    def __init__(self, embedding_map: dict[str, list[float]] | None = None):
        self._embedding_map = embedding_map or {}
        self.ingested: list[tuple[list[str], list[dict]]] = []

    def _compute_embeddings(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            # 命中显式映射优先, 否则用 hash 分桶到固定向量
            if t in self._embedding_map:
                out.append(self._embedding_map[t])
            else:
                # ponytail: 简单 hash → 固定向量, 让相同文本同向.
                # 升级路径: 真实 sentence transformer.
                h = abs(hash(t)) % 8
                vec = [0.0] * 8
                vec[h] = 1.0
                out.append(vec)
        return out

    def ingest(self, documents, metadatas=None, ids=None):
        self.ingested.append((documents, metadatas or []))

    def search(self, query, top_k=5):
        return []


@pytest.fixture
def tmp_memory():
    with tempfile.TemporaryDirectory() as tmp:
        # 默认 enable_semantic=True 但 vector_store=None → _enable_semantic=False
        # store/retrieve 走 FTS5 fallback, 不调 vector_store
        mem = LongTermMemory(db_path=Path(tmp) / "memory.db")
        yield mem


def _force_semantic(mem: LongTermMemory, vs: _MockVectorStore) -> None:
    """手动注入 mock vector_store, 让 cluster_memories 走 embedding 路径."""
    mem._vector_store = vs
    mem._enable_semantic = True


# === 场景 1: env=0 时 maintenance 跳过 cluster step ===

def test_env_off_maintenance_skips_cluster(tmp_memory):
    """HUGINN_MEMORY_CLUSTER 未设或=0 时, maintenance(cluster=True) 不调 cluster."""
    tmp_memory.store("memory A", category="fact")
    tmp_memory.store("memory B", category="fact")

    async def llm_fn(prompt: str) -> str:
        pytest.fail("env=0 时不应调 LLM")

    with patch.dict(os.environ, {"HUGINN_MEMORY_CLUSTER": "0"}):
        result = tmp_memory.maintenance(cluster=True, llm_chat_fn=llm_fn)

    assert result.get("clustered", 0) == 0
    assert result.get("archived", 0) == 0


# === 场景 2: archived=0 默认行为: 所有 alive memory 可 retrieve ===

def test_default_archived_zero_all_recallable(tmp_memory):
    """老条目 archived default 0, 全部可 retrieve, 行为不变."""
    tmp_memory.store("GaN band gap 3.4 eV", category="fact")
    tmp_memory.store("Si thermal conductivity 150 W/mK", category="fact")

    ga = tmp_memory.retrieve("GaN band gap")
    si = tmp_memory.retrieve("Si thermal")
    assert len(ga) >= 1
    assert len(si) >= 1


# === 场景 3: archived=1 时 retrieve 自动过滤 ===

def test_archived_entries_filtered_from_retrieve(tmp_memory):
    """update_archived(mid, True) 后, retrieve 不再返回该条目."""
    mid = tmp_memory.store("To be archived", category="fact", importance=0.9)
    tmp_memory.store("Keep alive", category="fact", importance=0.9)

    # archive 前都能查到
    before = tmp_memory.retrieve("archived")
    assert any(r["id"] == mid for r in before)

    ok = tmp_memory.update_archived(mid, archived=True)
    assert ok is True

    after = tmp_memory.retrieve("archived")
    assert all(r["id"] != mid for r in after), "archived=1 的条目不应被 retrieve 返回"


# === 场景 4: cluster_memories + compress_clusters 端到端 ===

def test_cluster_compress_end_to_end(tmp_memory):
    """3 条同主题 + 2 条另一主题, cluster 后压缩, archived 条目不再可检索."""
    # 显式 embedding map: 同主题同向, 异主题异向, cosine 相似度可控
    vs = _MockVectorStore(embedding_map={
        "GaN band gap is 3.4 eV": [1.0, 0.0, 0.0],
        "GaN has a band gap of 3.4 eV": [1.0, 0.0, 0.0],
        "The band gap of GaN equals 3.4 eV": [1.0, 0.0, 0.0],
        "Si thermal conductivity 150 W/mK": [0.0, 1.0, 0.0],
        "Silicon thermal cond. about 150 W/mK": [0.0, 1.0, 0.0],
    })
    _force_semantic(tmp_memory, vs)

    ids_gan = [
        tmp_memory.store("GaN band gap is 3.4 eV", category="fact", importance=0.7),
        tmp_memory.store("GaN has a band gap of 3.4 eV", category="fact", importance=0.6),
        tmp_memory.store("The band gap of GaN equals 3.4 eV", category="fact", importance=0.8),
    ]
    ids_si = [
        tmp_memory.store("Si thermal conductivity 150 W/mK", category="fact", importance=0.6),
        tmp_memory.store("Silicon thermal cond. about 150 W/mK", category="fact", importance=0.5),
    ]

    clusters = cluster_memories(tmp_memory, threshold=0.85)
    assert len(clusters) == 2, f"应有 2 cluster, got {len(clusters)}"
    all_clustered = {mid for c in clusters for mid in c}
    assert all_clustered == set(ids_gan) | set(ids_si)

    async def llm_fn(prompt: str) -> str:
        # 简单 deterministic 返回, 让 compress_clusters 写回 summary
        if "GaN" in prompt:
            return "GaN has a band gap of 3.4 eV (consolidated from 3 entries)."
        return "Si thermal conductivity ~150 W/mK (consolidated from 2 entries)."

    counts = compress_clusters(tmp_memory, clusters, llm_fn)
    assert counts["summarized"] == 2, f"应压缩 2 cluster, got {counts}"
    assert counts["archived"] == 5, f"应归档 5 原条目, got {counts}"

    # 原条目 archived=1, 不再被 retrieve 返回
    for mid in all_clustered:
        results = tmp_memory.retrieve(mid)  # 用 id 当 query, FTS5 不会命中
        assert all(r["id"] != mid for r in results), \
            f"archived 条目 {mid} 不应再被 retrieve 返回"

    # summary 写回 long-tier, 可被 retrieve 命中
    consolidated = tmp_memory.retrieve("consolidated")
    assert len(consolidated) >= 2, "应有 2 条 summary"


# === 场景 5: env=1 + maintenance 闭环 ===

def test_maintenance_with_cluster_env_on(tmp_memory):
    """HUGINN_MEMORY_CLUSTER=1 时, maintenance(cluster=True, llm_chat_fn) 真实闭环."""
    vs = _MockVectorStore(embedding_map={
        "fact A about X": [1.0, 0.0],
        "fact A prime about X": [1.0, 0.0],
        "fact A second about X": [1.0, 0.0],
    })
    _force_semantic(tmp_memory, vs)

    tmp_memory.store("fact A about X", category="fact")
    tmp_memory.store("fact A prime about X", category="fact")
    tmp_memory.store("fact A second about X", category="fact")

    async def llm_fn(prompt: str) -> str:
        return "Consolidated: X is characterized by A (3 sources merged)."

    with patch.dict(os.environ, {"HUGINN_MEMORY_CLUSTER": "1"}):
        result = tmp_memory.maintenance(cluster=True, llm_chat_fn=llm_fn)

    assert result.get("clustered", 0) >= 1, "maintenance 应触发 cluster step"
    assert result.get("archived", 0) >= 2, "原条目应被归档"
