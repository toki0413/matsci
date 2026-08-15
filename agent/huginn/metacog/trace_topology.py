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
from typing import Any

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


# jieba 懒加载缓存: None=未尝试, False=不可用, 否则为 jieba 模块.
# 中文拓扑证据语义重叠依赖中文分词, 复用 RAG 升级引入的 jieba.
_JIEBA: Any | None = None


def _get_jieba() -> Any | None:
    """懒加载 jieba. 首次尝试后缓存结果, 避免每次 _tokenize 都 import."""
    global _JIEBA
    if _JIEBA is None:
        try:
            import jieba
            _JIEBA = jieba
        except Exception:
            _JIEBA = False
    return _JIEBA if _JIEBA else None


def _tokenize(text: str) -> list[str]:
    """分词: 优先 jieba 中文分词, 否则 ASCII 字母数字 token. 不引外部硬依赖."""
    if not text:
        return []
    text_l = text.lower()
    jieba = _get_jieba()
    if jieba is not None:
        tokens = []
        seen = set()
        for tok in jieba.cut(text_l):
            if not tok or tok.isspace():
                continue
            if tok not in seen:
                seen.add(tok)
                tokens.append(tok)
        # 兜底: 保留纯字母数字 token (jieba 可能漏切的英文/数字)
        for tok in re.findall(r"[a-z0-9]+", text_l):
            if tok not in seen:
                seen.add(tok)
                tokens.append(tok)
        return tokens
    return re.findall(r"[a-z0-9]+", text_l)


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


# ---------- v15 schema 扩展 (Phase 2) ----------
#
# v14 entry 是 plain dict, v15 加 4 个新字段. 不引 dataclass — 跟 v14 形态一致,
# 升级路径是后续 phase 统一. ponytail: dict + helper 函数够用.
_V15_ENTRY_DEFAULTS = {
    "hypothesis_id": None,        # 关联到 HypothesisManifold 里的一个 hypothesis
    "log_posterior": 0.0,         # Bayesian log posterior (可负)
    "fisher_info": 0.0,           # Fisher metric 在该 hypothesis 处的 trace (≥0)
    "imagination_parent": None,   # 如果是 imagination 生成的新 h, 记录父 hypothesis_id
}


def upgrade_entry(entry: dict) -> dict:
    """补 v15 新字段到 entry, 原地修改. 旧 meta_trace.jsonl 加载时调用.

    旧 v14 entry 没有这 4 个字段, 自动补默认值. 同时若 darwin_score 缺失, 从
    log_posterior 推出 (v15 语义). 已存在的字段不覆盖 — v14 的 darwin_score 保留.

    Returns: 同一个 entry (方便链式调用).
    """
    for k, v in _V15_ENTRY_DEFAULTS.items():
        entry.setdefault(k, v)
    # v14 darwin_score 保留, 缺失时从 log_posterior 重算 (v15 语义)
    if "darwin_score" not in entry:
        entry["darwin_score"] = darwin_score_normalized(entry)
    return entry


def darwin_score_normalized(entry: dict) -> float:
    """v15: darwin_score = exp(log_posterior), -inf → 0.0.

    v14 的 darwin_score 是 [0,1] 适应度. v15 升级为 exp(log_posterior) 的归一化
    形式 (prototype 里 log_prior + log_lik 都 ≤0, 所以 exp 后 ∈ (0, 1], 跟 v14
    适配度尺度一致). 旧代码读 darwin_score 不破坏, 新代码应该用 log_posterior
    或这个归一化形式.

    ponytail: 单 entry 计算, 不做跨 hypothesis 归一化 (那需要全 manifold 的
    logsumexp). 升级路径: HypothesisManifold.posterior(obs) 给真正的归一化.
    这里 exp() 是概率尺度 ↔ log 尺度的桥.
    """
    lp = float(entry.get("log_posterior", 0.0) or 0.0)
    if lp == float("-inf"):
        return 0.0
    try:
        return math.exp(lp)
    except OverflowError:
        # log_posterior > ~709 — 非典型 (prototype 里 log_lik ≤ 0). 截断到 1.0.
        # ponytail: 真正归一化需要除以 sum(exp(log_post_i)) 跨所有 entry, 这里
        # 单 entry 截断够用, 升级路径是 posterior(obs).
        return 1.0


# ---------- Self-check ----------

def _selfcheck() -> None:
    """v15 schema 扩展自检.

    验证: 新字段默认值 / 旧 v14 entry 自动补默认值 / darwin_score_normalized 跟
    log_posterior 一致. 不引框架, 纯 assert.
    """
    # Test 1: 新字段默认值正确
    assert _V15_ENTRY_DEFAULTS["hypothesis_id"] is None
    assert _V15_ENTRY_DEFAULTS["log_posterior"] == 0.0
    assert _V15_ENTRY_DEFAULTS["fisher_info"] == 0.0
    assert _V15_ENTRY_DEFAULTS["imagination_parent"] is None

    # Test 2: 旧 v14 entry (没新字段) 自动补默认值, 旧字段保留
    v14_entry = {
        "simplex_id": "s1",
        "attempted": "compute band structure",
        "evidence": "band gap = 1.2 eV",
        "darwin_score": 0.5,
        "supported_ratio": 0.8,
        "cochain_type": "gradient",
    }
    upgrade_entry(v14_entry)
    assert v14_entry["hypothesis_id"] is None
    assert v14_entry["log_posterior"] == 0.0
    assert v14_entry["fisher_info"] == 0.0
    assert v14_entry["imagination_parent"] is None
    # v14 的 darwin_score 必须保留, 不能被覆盖
    assert v14_entry["darwin_score"] == 0.5, "v14 darwin_score 必须保留"
    # v14 其他字段也不能丢
    assert v14_entry["cochain_type"] == "gradient"
    assert v14_entry["supported_ratio"] == 0.8
    assert v14_entry["simplex_id"] == "s1"

    # Test 3: 新 v15 entry (有 log_posterior, 没 darwin_score) 自动补 darwin_score
    v15_entry = {
        "simplex_id": "s2",
        "hypothesis_id": "h_gr",
        "log_posterior": -0.5,
        "fisher_info": 0.3,
        "cochain_type": "gradient",
    }
    upgrade_entry(v15_entry)
    # darwin_score 应该从 log_posterior 推出: exp(-0.5) ≈ 0.6065
    expected = math.exp(-0.5)
    assert abs(v15_entry["darwin_score"] - expected) < 1e-9, (
        f"darwin_score should be exp(log_posterior)={expected}, "
        f"got {v15_entry['darwin_score']}"
    )
    # imagination_parent 缺失, 自动补 None
    assert v15_entry["imagination_parent"] is None

    # Test 4: darwin_score_normalized 跟 log_posterior 一致
    assert darwin_score_normalized({"log_posterior": 0.0}) == 1.0
    assert abs(darwin_score_normalized({"log_posterior": -1.0}) - math.exp(-1.0)) < 1e-9
    assert darwin_score_normalized({"log_posterior": float("-inf")}) == 0.0
    # 缺 log_posterior 默认 0.0 → exp(0) = 1.0 (跟 _V15_ENTRY_DEFAULTS 一致)
    assert darwin_score_normalized({}) == 1.0

    # Test 5: -inf log_posterior 边界 — upgrade_entry 应给出 darwin_score=0.0
    edge = {"simplex_id": "s3", "log_posterior": float("-inf")}
    upgrade_entry(edge)
    assert edge["darwin_score"] == 0.0, "exp(-inf) should be 0.0"
    assert edge["hypothesis_id"] is None

    # Test 6: imagination_parent 链 — 父 entry 跟子 entry 的关联
    parent_entry = {
        "simplex_id": "s4",
        "hypothesis_id": "h_newton",
        "log_posterior": -0.2,
    }
    child_entry = {
        "simplex_id": "s5",
        "hypothesis_id": "h_gr_variant",
        "log_posterior": -0.3,
        "imagination_parent": "h_newton",
    }
    upgrade_entry(parent_entry)
    upgrade_entry(child_entry)
    assert child_entry["imagination_parent"] == "h_newton"
    assert parent_entry["imagination_parent"] is None  # 父不是 imagination 生成的

    # Test 7: compute_betti 在 v15 entry 上仍能工作 (字段不冲突)
    entries = [
        {"simplex_id": "n1", "attempted": "test a", "evidence": "see a",
         "hypothesis_id": "h1", "log_posterior": -0.1, "fisher_info": 0.1},
        {"simplex_id": "n2", "attempted": "test b", "evidence": "see b",
         "hypothesis_id": "h2", "log_posterior": -0.2, "fisher_info": 0.2},
    ]
    for e in entries:
        upgrade_entry(e)
    b0, b1 = compute_betti(entries)
    assert b0 >= 1, f"β_0 should be ≥1, got {b0}"

    print("✓ trace_topology v15 schema self-check passed")
    print(f"  v14 entry upgraded: hypothesis_id={v14_entry['hypothesis_id']}, "
          f"log_posterior={v14_entry['log_posterior']}, "
          f"darwin_score preserved={v14_entry['darwin_score']}")
    print(f"  v15 entry: log_posterior={v15_entry['log_posterior']}, "
          f"darwin_score=exp(log_posterior)={v15_entry['darwin_score']:.4f}")
    print(f"  darwin_score_normalized(-inf) = {darwin_score_normalized(edge)}")
    print(f"  compute_betti on v15 entries: β_0={b0}, β_1={b1}")


if __name__ == "__main__":
    _selfcheck()
