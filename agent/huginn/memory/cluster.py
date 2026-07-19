"""P5 Memory Semantic Cluster & Compress.

把同主题近义条目聚成 cluster, LLM 浓缩成 1 条写回 long-tier,
原条目降级到 short-tier (TTL 6h) 自然 decay, 不破坏 schema.

设计 (ponytail):
- 复用 LongTermMemory._vector_store 的 embedding API, 不新建组件
- union-find 聚类, O(N^2) 相似度矩阵, N≤50 够用
- 默认关, 接入由 Task 2 做
- 异常吞掉要 log, 不隐藏 bug

升级路径:
- online cluster: 写时合并 (一条新 memory 进来直接找最近邻 cluster 合并),
  避免 batch 全量重算. 阈值动态调整由 decay policy 兜底.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Awaitable, Callable

import numpy as np

logger = logging.getLogger(__name__)

LLMChatFn = Callable[[str], Awaitable[str]]


def cluster_memories(
    ltm: Any,
    threshold: float = 0.85,
    top_n_per_query: int = 50,
) -> list[list[str]]:
    """把最近 alive memory 按语义相似度聚成 cluster.

    Args:
        ltm: LongTermMemory 实例
        threshold: cosine 相似度阈值, ≥ 此值归为同 cluster
        top_n_per_query: 取最近 N 条 alive memory 做聚类

    Returns:
        cluster 列表, 每个是 memory id list (string). 单条不返回.
        semantic 关闭 / embedding 不可用时返回空 list.
    """
    if not getattr(ltm, "_enable_semantic", False):
        logger.warning("cluster_memories: semantic disabled, skip")
        return []

    vs = getattr(ltm, "_vector_store", None)
    if vs is None:
        logger.warning("cluster_memories: no _vector_store, skip")
        return []

    # 拉最近 N 条 alive memory (跟 longterm._where_alive 一致)
    now = datetime.now().isoformat()
    with ltm._connect() as conn:
        rows = conn.execute(
            "SELECT id, content FROM memories "
            "WHERE expires_at IS NULL OR expires_at > ? "
            "ORDER BY created_at DESC LIMIT ?",
            (now, top_n_per_query),
        ).fetchall()

    if len(rows) < 2:
        return []

    ids = [r["id"] for r in rows]
    contents = [r["content"] for r in rows]

    # 复用 vector_store 的 embedding API (不新建 embedding 组件)
    embeddings = vs._compute_embeddings(contents)
    if not embeddings or len(embeddings) != len(ids):
        logger.warning(
            "cluster_memories: embedding failed (%d/%d), skip",
            len(embeddings) if embeddings else 0, len(ids),
        )
        return []

    # cosine 相似度矩阵: normed @ normed.T
    emb = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # 防 0 向量除零
    normed = emb / norms
    sim = normed @ normed.T

    # union-find (path compression)
    parent = list(range(len(ids)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # ponytail: O(N^2) 双层循环, N≤50 才 2500 次比较, 够用.
    # 升级路径: 上三角扫描 + HNSW 索引避免全量比对.
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if sim[i, j] >= threshold:
                union(i, j)

    # 收集 cluster, 单条不返回
    groups: dict[int, list[str]] = {}
    for i, mid in enumerate(ids):
        groups.setdefault(find(i), []).append(mid)

    return [g for g in groups.values() if len(g) >= 2]


def summarize_cluster(
    ltm: Any,
    ids: list[str],
    llm_chat_fn: LLMChatFn,
) -> dict | None:
    """LLM 合并 cluster 内 N 条 memory 成 1 条浓缩摘要.

    Args:
        ltm: LongTermMemory 实例
        ids: cluster 内 memory id 列表 (string, mem_YYYYMMDD_HHMMSS_xxxxxxxx)
        llm_chat_fn: async, 输入 prompt 返回 LLM 文本

    Returns:
        {summary, importance, source, tags, category, cluster_ids}, 失败 None
    """
    if not ids:
        return None

    with ltm._connect() as conn:
        placeholders = ",".join("?" * len(ids))
        rows = conn.execute(
            f"SELECT id, content, importance, source, created_at, tags, category "
            f"FROM memories WHERE id IN ({placeholders})",
            tuple(ids),
        ).fetchall()

    if not rows:
        return None

    # 拼 prompt (附 metadata 让 LLM 知道每条来源/时间)
    items = []
    for r in rows:
        items.append(
            f"[{r['id']}] (importance={r['importance']}, source={r['source']}, "
            f"created={r['created_at']}, category={r['category']})\n{r['content']}"
        )
    prompt = (
        f"把以下 {len(rows)} 条同主题 memory 合并成 1 条浓缩摘要, "
        f"保留 evidence/source/timestamp 关键信息。输出格式:纯文本摘要,不超过 500 词。\n\n"
        + "\n---\n".join(items)
    )

    # ponytail: 同步包装 async LLM. 不能在已有 event loop 里调,
    # 升级路径: 整体改 async 或用 run_coroutine_threadsafe.
    try:
        summary = asyncio.run(llm_chat_fn(prompt))
    except Exception:
        logger.warning("summarize_cluster: LLM call failed", exc_info=True)
        return None

    if not summary or not summary.strip():
        return None

    importance = max((float(r["importance"]) for r in rows), default=0.5)
    sources = [r["source"] for r in rows if r["source"]]
    source = " | ".join(sources) if sources else ""
    cats = [r["category"] for r in rows if r["category"]]
    # ponytail: 多数投票用 Counter, 平票取排第一的. 升级路径: 按 importance 加权.
    category = Counter(cats).most_common(1)[0][0] if cats else "fact"
    tags = ["cluster_summary"] + [r["id"] for r in rows]

    return {
        "summary": summary.strip(),
        "importance": importance,
        "source": source,
        "tags": tags,
        "category": category,
        "cluster_ids": [r["id"] for r in rows],
    }


def compress_clusters(
    ltm: Any,
    clusters: list[list[str]],
    llm_chat_fn: LLMChatFn,
) -> dict[str, int]:
    """对每个 cluster 调 summarize_cluster, 写回 summary, 原条目降级.

    summary 写成 long-tier (永久保留). 原条目 tier 改 short (TTL 6h) +
    tags 追加 archived_cluster_{new_id} 标记. 不动 schema, Task 2 接入时
    再加正式 archived 字段.

    Returns:
        {summarized, archived, skipped, failed}
    """
    counts = {"summarized": 0, "archived": 0, "skipped": 0, "failed": 0}

    for cluster in clusters:
        if len(cluster) < 2:
            counts["skipped"] += 1
            continue

        result = summarize_cluster(ltm, cluster, llm_chat_fn)
        if result is None:
            # LLM 失败 → skip 这个 cluster, 不算 failed
            counts["skipped"] += 1
            continue

        cluster_tag = f"cluster_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            new_id = ltm.store(
                content=result["summary"],
                category=result["category"],
                tags=result["tags"] + [cluster_tag],
                source=result["source"],
                importance=result["importance"],
                tier="long",
            )
        except Exception:
            logger.warning("compress_clusters: store summary failed", exc_info=True)
            counts["failed"] += 1
            continue

        # 原条目标记 archived (Task 2 接入: longterm.py 加了 archived 字段)
        # archived=1 → _where_alive 自动过滤, 不参与 retrieve/recall.
        # 保留在表里, 留档可 rollback 或 audit.
        # ponytail: 每条单独 update_archived, N≤50 不值得批量化.
        # 升级路径: 一次 UPDATE ... WHERE id IN (...).
        for mid in result["cluster_ids"]:
            try:
                ok = ltm.update_archived(mid, archived=True)
                if ok:
                    counts["archived"] += 1
            except Exception:
                logger.warning(
                    "compress_clusters: archive %s failed", mid, exc_info=True
                )

        counts["summarized"] += 1

    return counts


# === 自检 (不依赖真实 SQLite/vector_store, 用 SimpleNamespace mock) ===


class _FakeRow(dict):
    """sqlite3.Row 替身, 支持 row['col'] 索引."""


class _FakeConn:
    """按 SQL 关键词匹配返回预制 rows. 不真正执行 SQL."""

    def __init__(self, rows_by_pattern: dict[str, list[_FakeRow]]):
        self._rows = rows_by_pattern

    def execute(self, sql: str, params: tuple = ()):  # noqa: ARG002
        for pattern, rows in self._rows.items():
            if pattern in sql:
                return _FakeCursor(rows)
        return _FakeCursor([])

    def commit(self):  # noqa: ARG002
        pass


class _FakeCursor:
    def __init__(self, rows: list):
        self._rows = rows

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


def _make_mock_ltm(
    rows: list[_FakeRow],
    embeddings: list[list[float]] | None,
    enable_semantic: bool = True,
    store_id: str = "mem_new_xxxxxxxx",
) -> SimpleNamespace:
    """构造 mock LongTermMemory 用于自检."""
    # 三个 pattern 互不重叠: expires_at (cluster_memories),
    # id IN (summarize_cluster), SELECT tags (compress archive)
    fake_conn = _FakeConn({
        "expires_at": rows,
        "id IN": rows,
        "SELECT tags": rows,
    })

    @contextmanager
    def fake_connect():
        yield fake_conn

    def fake_store(**kwargs):  # noqa: ARG001
        return store_id

    def fake_update(entry_id, **kwargs):  # noqa: ARG001
        return True

    vs = SimpleNamespace(
        _compute_embeddings=lambda texts: embeddings,
    )

    return SimpleNamespace(
        _enable_semantic=enable_semantic,
        _vector_store=vs,
        _connect=fake_connect,
        store=fake_store,
        update=fake_update,
    )


def _row(
    id_: str,
    content: str,
    importance: float = 0.5,
    source: str = "",
    created_at: str = "2026-01-01T00:00:00",
    tags: list[str] | None = None,
    category: str = "fact",
) -> _FakeRow:
    return _FakeRow({
        "id": id_, "content": content, "importance": importance,
        "source": source, "created_at": created_at,
        "tags": json.dumps(tags or []), "category": category,
    })


if __name__ == "__main__":
    # 场景 1: semantic 关闭 → 返回空 list
    ltm = _make_mock_ltm([], None, enable_semantic=False)
    assert cluster_memories(ltm) == [], "semantic 关闭应返回空 list"
    print("1. semantic off fallback OK")

    # 场景 2: 单 cluster (3 条近义, embedding 同向 → cosine=1.0)
    rows = [
        _row("mem_a", "GaN band gap is 3.4 eV", 0.7, "vasp_calc:GaN"),
        _row("mem_b", "GaN has a band gap of 3.4 eV", 0.6, "vasp_calc:GaN"),
        _row("mem_c", "The band gap of GaN equals 3.4 eV", 0.8, "vasp_calc:GaN"),
    ]
    emb = [[1.0, 0.0, 0.0]] * 3  # 完全相同 → cosine=1.0
    ltm = _make_mock_ltm(rows, emb)
    clusters = cluster_memories(ltm)
    assert len(clusters) == 1, f"应聚成 1 cluster, got {len(clusters)}"
    assert set(clusters[0]) == {"mem_a", "mem_b", "mem_c"}
    print(f"2. single cluster OK ({len(clusters[0])} ids)")

    # 场景 3: 多 cluster (6 条, 2 组近义, 每组 3 条)
    rows = [
        _row("mem_a", "GaN band gap 3.4 eV"),
        _row("mem_b", "GaN bandgap = 3.4 eV"),
        _row("mem_c", "GaN has band gap 3.4 eV"),
        _row("mem_d", "Si thermal conductivity 150 W/mK"),
        _row("mem_e", "Silicon thermal cond. ~150 W/mK"),
        _row("mem_f", "Si thermal conduct. about 150 W/mK"),
    ]
    emb = [
        [1.0, 0.0], [1.0, 0.0], [1.0, 0.0],  # 组 1: GaN
        [0.0, 1.0], [0.0, 1.0], [0.0, 1.0],  # 组 2: Si
    ]
    ltm = _make_mock_ltm(rows, emb)
    clusters = cluster_memories(ltm)
    assert len(clusters) == 2, f"应聚成 2 cluster, got {len(clusters)}"
    assert all(len(c) == 3 for c in clusters), "每个 cluster 应有 3 个 id"
    print(f"3. multi cluster OK ({len(clusters)} clusters)")

    # 场景 4: singleton skip (1 条 memory → 返回空 list)
    rows = [_row("mem_solo", "lone memory")]
    ltm = _make_mock_ltm(rows, [[1.0, 0.0]])
    clusters = cluster_memories(ltm)
    assert clusters == [], "单条应返回空 list"
    print("4. singleton skip OK")

    # 场景 5: LLM 失败 → compress_clusters skipped=1, summarized=0
    rows = [
        _row("mem_x", "GaN band gap 3.4 eV"),
        _row("mem_y", "GaN bandgap = 3.4 eV"),
    ]
    emb = [[1.0, 0.0], [1.0, 0.0]]
    ltm = _make_mock_ltm(rows, emb)
    clusters = cluster_memories(ltm)
    assert len(clusters) == 1

    async def failing_llm(prompt: str) -> str:  # noqa: ARG001
        raise RuntimeError("LLM unavailable")

    counts = compress_clusters(ltm, clusters, failing_llm)
    assert counts["summarized"] == 0, f"summarized 应为 0, got {counts}"
    assert counts["skipped"] == 1, f"skipped 应为 1, got {counts}"
    print(f"5. LLM fail skip OK (counts={counts})")

    print("cluster selfcheck All passed (5 scenarios)")
