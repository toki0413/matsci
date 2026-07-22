"""Visual hippocampus — 跨 session 视觉记忆累积 + 时间衰减 + 检索.

借鉴 Ego3D-VLM (arXiv:2509.06266, Huawei, 2025-09) 海马体 place/grid cells
理论: 视觉记忆不是瞬时帧, 是累积 + 衰减 + 检索的动态过程. QW1 在 EngineState
加了 `_visual_primitives_history` 字段, 本模块在这上面构建 hippocampus API:

  record(primitives, session_id) -> 累积一条记忆 (JSON string 入 list)
  recall(query, top_k, decay)    -> 检索 top_k 历史 primitives (含时间衰减)
  forget(max_age_s)               -> 清除超期记忆

跟 image_index (QW4) 集成: 当 entry 带 image_id 时, recall 也能按 image_id
反查 embeddings (state._image_embeddings).

设计原则:
  - decoupling: 函数接受 list[str] (history) 而不是 EngineState, 更通用
  - ponytail: 关键词 tf-idf 匹配, 不上 embedding model
  - ceiling: text-only primitives, 不存 raw base64. 升级路径接 image_index.

接入点:
  - autoloop/visual_inspect.py 每次 visual_inspect 后调 record
  - runtime/engine_state.py save/load 已经自动持久化 _visual_primitives_history
  - env flag HUGINN_USE_HIPPOCAMPUS=1 才开 (默认 off)
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

_HIPPOCAMPUS_FLAG = "HUGINN_USE_HIPPOCAMPUS"
_DEFAULT_TAU_S = 3600.0  # 衰减时间常数 (s), 默认 1 小时半衰
_DEFAULT_MAX_HISTORY = 100  # 短期记忆容量上限 (LRU 截断)


def use_hippocampus() -> bool:
    """HUGINN_USE_HIPPOCAMPUS=1 才开. 默认 off."""
    return os.environ.get(_HIPPOCAMPUS_FLAG, "0") == "1"


# ── record ─────────────────────────────────────────────────────────────────


def record(
    history: list[str],
    primitives: str,
    session_id: str = "",
    ts: float | None = None,
    image_id: str | None = None,
    max_history: int = _DEFAULT_MAX_HISTORY,
) -> str:
    """累积一条视觉记忆到 history (in-place append, 返回 entry JSON).

    entry 格式:
      {"ts": float, "session_id": str, "primitives": str, "image_id"?: str}

    ponytail: 直接 list.append + 截断, 不上 ring buffer. LRU 策略 = 丢最旧.
    ceiling: 不去重, 同一 primitives 多次记录都保留. 升级路径: dedup by hash.

    Args:
        history: state._visual_primitives_history (in-place 修改)
        primitives: text primitives string (e.g. "[bands] peak=...")
        session_id: 来源 session 标识
        ts: 时间戳 (None = time.time())
        image_id: 关联图像 ID (可选, 用于反查 embeddings)
        max_history: 容量上限, 超过截断 (默认 100)

    Returns:
        entry JSON string (也 append 到 history)
    """
    entry: dict[str, Any] = {
        "ts": float(ts) if ts is not None else time.time(),
        "session_id": session_id,
        "primitives": primitives,
    }
    if image_id:
        entry["image_id"] = image_id
    line = json.dumps(entry, ensure_ascii=False)
    history.append(line)
    # LRU 截断: 丢最旧的 (不是真正 LRU, 是 FIFO; 升级路径: 按访问频率)
    if len(history) > max_history:
        del history[: len(history) - max_history]
    return line


# ── recall ─────────────────────────────────────────────────────────────────


def _tokenize(text: str) -> list[str]:
    """简单分词: 小写 + 非字母数字切分. ponytail: 不上 jieba/spacy."""
    out: list[str] = []
    cur = []
    for ch in text.lower():
        if ch.isalnum():
            cur.append(ch)
        else:
            if cur:
                out.append("".join(cur))
                cur = []
    if cur:
        out.append("".join(cur))
    return out


def _tfidf_score(query_tokens: list[str], doc: str) -> float:
    """简单 tf 匹配 (不真算 idf, ponytail). 返回 [0, 1] 归一化分数."""
    if not query_tokens or not doc:
        return 0.0
    doc_tokens = _tokenize(doc)
    if not doc_tokens:
        return 0.0
    doc_set = set(doc_tokens)
    hits = sum(1 for t in query_tokens if t in doc_set)
    return hits / len(query_tokens)


def recall(
    history: list[str],
    query: str | None = None,
    text_query: str | None = None,
    top_k: int = 5,
    decay: bool = True,
    tau_s: float = _DEFAULT_TAU_S,
    now_ts: float | None = None,
    min_score: float = 0.0,
) -> list[dict[str, Any]]:
    """检索历史 primitives, 返回 top_k 条按综合分数排序.

    综合分数 = text_match_score * decay_factor
      - text_match_score: tf 匹配 (query 或 text_query), 无 query 时 = 1.0
      - decay_factor: exp(-Δt / tau_s), Δt = now - entry.ts

    ponytail: 不上 embedding 相似度, 用关键词 tf. 升级路径接 sentence-transformers.

    Args:
        history: state._visual_primitives_history
        query: 简单关键词 (兼容 QW4 image_index.search 的 query 参数)
        text_query: 同 query, 别名 (QW4 风格), 优先级高于 query
        top_k: 返回前 K 条
        decay: 是否启用时间衰减 (默认 True)
        tau_s: 衰减时间常数 (s), 默认 3600 (1h 半衰)
        now_ts: 当前时间戳 (None = time.time(), 测试可注入)
        min_score: 最小分数阈值 (默认 0, 不过滤)

    Returns:
        list[dict]: [{entry: dict, score, text_match, decay_factor, age_s}]
    """
    if not history:
        return []

    # text_query 优先, 否则用 query
    q = text_query if text_query is not None else query
    q_tokens = _tokenize(q) if q else []
    now = float(now_ts) if now_ts is not None else time.time()

    scored: list[dict[str, Any]] = []
    for line in history:
        try:
            entry = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        primitives = entry.get("primitives", "")
        ts = float(entry.get("ts", 0.0))

        # text match
        if q_tokens:
            text_match = _tfidf_score(q_tokens, primitives)
            if text_match <= 0.0:
                continue  # query 有 token 但无命中, 跳过
        else:
            text_match = 1.0  # 无 query 时所有条目等权

        # decay
        age_s = max(0.0, now - ts)
        if decay:
            decay_factor = math.exp(-age_s / tau_s) if tau_s > 0 else 1.0
        else:
            decay_factor = 1.0

        score = text_match * decay_factor
        if score < min_score:
            continue
        scored.append({
            "entry": entry,
            "score": score,
            "text_match": text_match,
            "decay_factor": decay_factor,
            "age_s": age_s,
        })

    # 排序: score 降序, 同分时 age_s 升序 (新优先)
    scored.sort(key=lambda x: (-x["score"], x["age_s"]))
    return scored[:top_k]


# ── forget ──────────────────────────────────────────────────────────────────


def forget(
    history: list[str],
    max_age_s: float | None = None,
    now_ts: float | None = None,
    min_score_threshold: float | None = None,
) -> int:
    """清除超期 / 低分记忆, 返回清除条数 (in-place 修改 history).

    两种策略 (可同时启用):
      - max_age_s: 删除 age > max_age_s 的条目
      - min_score_threshold: 删除 primitives 长度 < 阈值的条目 (低信息密度)

    ponytail: 直接 list comprehension 重建, 不上 LRU counter.
    ceiling: 不考虑访问频率, 只看 age / 长度. 升级路径接 access_count.

    Args:
        history: state._visual_primitives_history (in-place 修改)
        max_age_s: 最大 age (s), 超期删除
        now_ts: 当前时间戳 (None = time.time())
        min_score_threshold: primitives 字符长度阈值, 短于此删除

    Returns:
        清除的条数
    """
    if not history:
        return 0
    now = float(now_ts) if now_ts is not None else time.time()
    kept: list[str] = []
    removed = 0
    for line in history:
        try:
            entry = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            # 损坏的 entry 直接删
            removed += 1
            continue
        if max_age_s is not None:
            ts = float(entry.get("ts", 0.0))
            if (now - ts) > max_age_s:
                removed += 1
                continue
        if min_score_threshold is not None:
            prim = entry.get("primitives", "")
            if len(prim) < min_score_threshold:
                removed += 1
                continue
        kept.append(line)
    history[:] = kept
    return removed


# ── selfcheck ──────────────────────────────────────────────────────────────


def _selfcheck() -> None:
    """L13 selfcheck: record → recall → forget 全链路."""
    # 1. record 3 条 → history 长度 = 3
    h: list[str] = []
    record(h, "[bands] peak=2.5", session_id="s1", ts=100.0)
    record(h, "[lattice] d=4Å", session_id="s1", ts=200.0)
    record(h, "[particles] n=10 detected", session_id="s2", ts=300.0)
    assert len(h) == 3, f"expected 3, got {len(h)}"
    # 验证 entry 格式
    e1 = json.loads(h[0])
    assert e1["ts"] == 100.0 and e1["session_id"] == "s1" and "peak" in e1["primitives"]
    print(f"1. record 3 → len={len(h)}, e1.ts={e1['ts']}")

    # 2. recall 无 query → 返回 top_k (按 decay 排序, 最新优先)
    r2 = recall(h, top_k=3, now_ts=400.0, tau_s=3600.0)
    assert len(r2) == 3, f"expected 3, got {len(r2)}"
    # 最新 (ts=300) 应排第一
    assert r2[0]["entry"]["ts"] == 300.0, f"newest first expected, got ts={r2[0]['entry']['ts']}"
    # decay_factor < 1 (因为 age > 0)
    assert 0 < r2[0]["decay_factor"] < 1, f"decay_factor should be (0,1): {r2[0]['decay_factor']}"
    print(f"2. recall no query → top1 ts={r2[0]['entry']['ts']}, decay={r2[0]['decay_factor']:.3f}")

    # 3. recall text_query="lattice" → 只返回含 lattice 的
    r3 = recall(h, text_query="lattice", top_k=5, now_ts=400.0, decay=False)
    assert len(r3) == 1, f"expected 1 lattice match, got {len(r3)}"
    assert "lattice" in r3[0]["entry"]["primitives"]
    # decay=False → decay_factor=1.0
    assert r3[0]["decay_factor"] == 1.0
    print(f"3. recall text_query='lattice' → {len(r3)} match, score={r3[0]['text_match']:.2f}")

    # 4. decay 验证: 旧记录权重 < 新记录
    r4 = recall(h, top_k=3, now_ts=1e9, tau_s=100.0)  # now 非常大, 全部衰减很多
    # 全部 decay_factor 应该非常小
    assert all(s["decay_factor"] < 0.01 for s in r4), f"decay should be tiny: {r4}"
    # 但相对排序仍 ts=300 > ts=200 > ts=100
    ts_sorted = [s["entry"]["ts"] for s in r4]
    assert ts_sorted == [300.0, 200.0, 100.0], f"newest-first expected, got {ts_sorted}"
    _factors = [f"{s['decay_factor']:.4f}" for s in r4]
    print(f"4. decay large Δt → factors={_factors}")

    # 5. forget max_age_s=150 (now=400) → 删 ts=100, 200 (age>150), 留 ts=300
    h5 = list(h)  # 复制避免污染
    n_removed = forget(h5, max_age_s=150.0, now_ts=400.0)
    assert n_removed == 2, f"expected 2 removed, got {n_removed}"
    assert len(h5) == 1
    kept_ts = json.loads(h5[0])["ts"]
    assert kept_ts == 300.0, f"expected ts=300 kept, got {kept_ts}"
    print(f"5. forget max_age=150 → removed={n_removed}, kept ts={kept_ts}")

    # 6. LRU 截断: max_history=2 时 record 第 3 条会丢最旧
    h6: list[str] = []
    record(h6, "a", ts=1.0, max_history=2)
    record(h6, "b", ts=2.0, max_history=2)
    record(h6, "c", ts=3.0, max_history=2)
    assert len(h6) == 2, f"expected 2 after LRU, got {len(h6)}"
    ts6 = [json.loads(x)["ts"] for x in h6]
    assert ts6 == [2.0, 3.0], f"oldest dropped expected, got {ts6}"
    print(f"6. LRU max_history=2 → kept ts={ts6}")

    # 7. 空 history → recall 返回 []
    assert recall([], top_k=5) == []
    # 损坏 entry 被 recall 跳过
    h7 = ["not_json", json.dumps({"ts": 1.0, "primitives": "ok"})]
    r7 = recall(h7, top_k=5, decay=False)
    assert len(r7) == 1, f"corrupt entry skipped expected, got {len(r7)}"
    print(f"7. corrupt entry skipped → {len(r7)} valid")

    print("L13 ALL CHECKS PASSED")


if __name__ == "__main__":
    _selfcheck()
