"""Trace topology: Betti number computation (β_0 / β_1).

工程近似: 用 networkx 弱连通分量数 + cycle_basis 近似 simplicial homology 的
β_0/β_1. 不是完整 homology 计算 — 真正的 betti 需要边界矩阵 Smith normal
form (O(n^3)). ponytail: entry 数 ≤50 上限, 超过按 darwin_score 截断, 控制成本.

高阶网络视角 (spec §"Betti 数计算"): Meta-Trace 的 entry 是 0-simplex, 当
entry_i.attempted 跟 entry_j.evidence 语义重叠 > 0.7 时形成 1-simplex (i, j).
β_0 = 独立假设链数, β_1 = 循环回退路径数. β_1 > 0 解锁 Step3→Step2 回退 (拓扑许可).
"""
from __future__ import annotations

import math
import re
from collections import Counter

# 优先复用 context_builder._compute_semantic_overlap (Task 3 实现).
# Task 3 未完成时用本地 TF-IDF cosine 兜底 — 不阻塞 Task 4 self-check.
# ponytail: 升级路径是 Task 3 完成后 import 自动替换, 这里不动.
try:
    from huginn.context_builder import (
        _compute_semantic_overlap as _sem_overlap,
    )
except Exception:
    def _sem_overlap(a: str, b: str) -> float:
        return _local_tfidf_cosine(a, b)


def _tokenize(text: str) -> list[str]:
    """简单分词: 小写 + 提取字母数字 token. 不引外部依赖."""
    if not text:
        return []
    return re.findall(r"[a-z0-9]+", text.lower())


def _local_tfidf_cosine(a: str, b: str) -> float:
    """极简 TF-IDF + cosine — 跟 context_builder 待实现版本对齐.

    ponytail: 单文档无 corpus, IDF 退化为 1, 等价 TF-cosine. n≤50 上限下够用.
    升级路径: Task 3 完成后由 context_builder._compute_semantic_overlap 替换.
    """
    if not a or not b:
        return 0.0
    ta = _tokenize(a)
    tb = _tokenize(b)
    if not ta or not tb:
        return 0.0
    ca = Counter(ta)
    cb = Counter(tb)
    dot = sum(ca[t] * cb[t] for t in ca.keys() & cb.keys())
    na = math.sqrt(sum(v * v for v in ca.values()))
    nb = math.sqrt(sum(v * v for v in cb.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _evidence_to_str(entry: dict) -> str:
    """entry.evidence 可能是 list 或 str, 统一拍扁成 str."""
    ev = entry.get("evidence", "")
    if isinstance(ev, list):
        return " ".join(str(x) for x in ev)
    return str(ev) if ev else ""


def _truncate_by_darwin(entries: list, cap: int = 50) -> list:
    """entry 数 > cap 时按 darwin_score 降序取 top cap.

    ponytail: 简单排序 O(n log n), 不引堆. n≤几百时够用.
    """
    if len(entries) <= cap:
        return entries
    return sorted(
        entries,
        key=lambda e: float(e.get("darwin_score", 0.0) or 0.0),
        reverse=True,
    )[:cap]


def compute_betti(trace_entries: list) -> tuple[int, int]:
    """算 Meta-Trace 的 (β_0, β_1).

    建图: 每个 entry 是 vertex (用 simplex_id 作 node id). 当 entry_i.attempted
    跟 entry_j.evidence 语义重叠 > 0.7 时加一条 edge (i, j).

    β_0 = 弱连通分量数 (独立假设链数)
    β_1 = cycle_basis 环路数 (循环回退路径数)

    工程近似: networkx 弱连通 + cycle_basis 不是完整 simplicial homology.
    真正 homology 需要边界矩阵 Smith normal form (O(n^3)). ponytail: entry 数
    ≤50 上限, 超过按 darwin_score 截断, 控制 O(n^3) 成本. 升级路径: 引
    `gudhi` / `ripser` 算真正 persistent homology.

    Args:
        trace_entries: list of dict, 每个至少有 simplex_id / attempted / evidence.

    Returns:
        (β_0, β_1) tuple. networkx 缺失时 β_1 回退到 0 (保守估计, 真实环路
        可能漏报).
    """
    if not trace_entries:
        return (0, 0)

    entries = _truncate_by_darwin(trace_entries, cap=50)

    # 提取 node id (simplex_id 缺失用 index 兜底, 保证 vertex 唯一)
    nodes: list[str] = []
    for i, e in enumerate(entries):
        sid = e.get("simplex_id")
        nodes.append(sid if sid else f"node_{i}")

    # 建边: attempted_i 跟 evidence_j 重叠 > 0.7 → edge (i, j)
    edges: list[tuple[str, str]] = []
    threshold = 0.7
    for i, ei in enumerate(entries):
        att_i = str(ei.get("attempted", "") or "")
        if not att_i:
            continue
        for j, ej in enumerate(entries):
            if i == j:
                continue
            ev_j = _evidence_to_str(ej)
            if not ev_j:
                continue
            try:
                overlap = _sem_overlap(att_i, ev_j)
            except Exception:
                overlap = 0.0
            if overlap > threshold:
                edges.append((nodes[i], nodes[j]))

    try:
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        # 无向图: connected_components 等价弱连通分量
        beta_0 = nx.number_connected_components(G)
        beta_1 = len(nx.cycle_basis(G))
        return (beta_0, beta_1)
    except ImportError:
        # networkx 没装 — 并查集算 β_0, β_1 回退到 0 (保守).
        # ponytail: 上限是 β_1 保守估计, 真实环路可能漏报. 升级路径: pip install networkx.
        return (_beta_0_union_find(nodes, edges), 0)


def _beta_0_union_find(nodes: list, edges: list) -> int:
    """并查集算连通分量数 — networkx 缺失时的纯 stdlib fallback.

    ponytail: path compression + union by rank 省了, n≤50 不需要. 升级路径:
    上规模时换 networkx.
    """
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        if a in parent and b in parent:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

    roots = {find(n) for n in nodes}
    return len(roots)
