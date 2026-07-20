"""Trajectory success pattern extractor (P2).

不新建 skill_library 组件. 复用 KB (ChromaDB) + auto_ingest 路径, 在
trajectory 成功结束时调一次 LLM 抽 "可复用 pattern", 写入 KB. 下次任务
开始时 RAG 自然召回, LLM 自己决定何时复用.

参考:
- Voyager (Wang et al. 2023): skill library, 但我们要的是 in-context 复用,
  不是可执行 skill 缓存.
- Alita (AutoGPT-style): 抽 task-solving pattern.
- Awesome-Long-Horizon-Agents survey Pillar II Self-Evolution 章节.

设计原则 (ponytail):
- 不抽可执行 skill (要 sandbox 验证, 成本高)
- 不建专门 skill 索引 (ChromaDB 已是索引)
- 不建 skill dispatcher (LLM 自己决定何时复用)
- 只在 trajectory 成功时抽一次 (失败 trajectory 不抽, 避免污染)
- 单次 LLM 调用 (deepseek-chat, 跟 PRT Level 1 / PRM verifier 同款)

接入:
  from huginn.knowledge.trajectory_pattern import (
      extract_and_store_pattern,
  )
  if goal_achieved:
      extract_and_store_pattern(
          objective=objective,
          trajectory=trajectory_data,
          final_output=final_output,
          llm_chat_fn=llm_chat_fn,
      )

升级路径:
- 失败 trajectory 也抽 (但标 failure_lesson, 不混入 success pattern)
- 多次 trajectory 抽出来的 pattern 做 dedup + 合并
- pattern 加 confidence 字段, 被复用且成功时 +ε, 失败时 -ε
"""

from __future__ import annotations

import json
import logging
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


_PATTERN_PROMPT_TEMPLATE = """You are extracting a reusable problem-solving pattern from a successful agent trajectory.

The user's objective was:
{objective}

The agent achieved the goal. Extract a CONCISE reusable pattern that would help
a future agent solve similar problems. Output JSON only, no markdown fences:

{{
  "task_pattern": "1-line description of the problem type this pattern solves",
  "key_steps": ["step1 in 1 line", "step2 in 1 line", "..."],
  "key_decisions": ["decision1: why X not Y (1 line)", "decision2: ..."],
  "pitfalls": ["pitfall1 + how to avoid (1 line)"],
  "applicability": "when to use this pattern (1 line)"
}}

Rules:
- Keep each field under 200 chars.
- 3-7 key_steps, 2-5 key_decisions, 1-4 pitfalls.
- Focus on what's REUSABLE, not what's unique to this run.
- If trajectory is too short / no clear pattern, return {{"task_pattern": "", "key_steps": [], "key_decisions": [], "pitfalls": [], "applicability": ""}}.

Trajectory summary (tool_calls, phases, results):
{trajectory_summary}
"""


def _summarize_trajectory(trajectory: dict | None, final_output: str) -> str:
    """把 trajectory 压成 prompt 能塞的摘要.

    ponytail: 只抽 tool_calls + phases 名字, 不塞完整 args/result.
    升级路径: trajectory 长 (H3) 时换 LLM 摘要再喂.
    """
    if not trajectory:
        return f"(no trajectory data)\nFinal output: {final_output[:500]}"
    parts = []
    # phases
    phases = trajectory.get("phases") or trajectory.get("phase_history") or []
    if phases:
        phase_names = [p.get("name", str(p)) if isinstance(p, dict) else str(p)
                       for p in phases[:20]]
        parts.append(f"Phases: {', '.join(phase_names)}")
    # tool calls
    tc = trajectory.get("tool_calls") or []
    if tc:
        tool_names = []
        for c in tc[:30]:
            if isinstance(c, dict):
                tool_names.append(c.get("tool", c.get("name", "?")))
            else:
                tool_names.append(str(c))
        parts.append(f"Tool calls ({len(tc)} total): {', '.join(tool_names)}")
    # final output 截断
    parts.append(f"Final output: {final_output[:500]}")
    return "\n".join(parts)


def _build_pattern_text(parsed: dict, objective: str) -> str:
    """把 LLM 抽出来的 pattern dict 拼成可检索的文本."""
    if not parsed.get("task_pattern"):
        return ""
    parts = [f"REUSABLE PATTERN (from successful task: {objective[:120]})"]
    parts.append(f"Task type: {parsed.get('task_pattern', '')}")
    steps = parsed.get("key_steps") or []
    if steps:
        parts.append("Key steps:")
        for i, s in enumerate(steps, 1):
            parts.append(f"  {i}. {s}")
    decisions = parsed.get("key_decisions") or []
    if decisions:
        parts.append("Key decisions:")
        for d in decisions:
            parts.append(f"  - {d}")
    pitfalls = parsed.get("pitfalls") or []
    if pitfalls:
        parts.append("Pitfalls:")
        for p in pitfalls:
            parts.append(f"  - {p}")
    appl = parsed.get("applicability", "")
    if appl:
        parts.append(f"When to use: {appl}")
    return "\n".join(parts)


def _parse_pattern_response(resp: str) -> dict | None:
    """解析 LLM 返回的 pattern JSON. 失败返回 None."""
    if not resp:
        return None
    resp = resp.strip()
    if resp.startswith("```"):
        resp = resp.strip("`")
        if resp.lower().startswith("json"):
            resp = resp[4:]
    start = resp.find("{")
    end = resp.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(resp[start:end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    # 空 pattern (LLM 判定无可复用内容) → 返回空 dict 标记
    if not data.get("task_pattern"):
        return {}
    # 字段类型清洗
    cleaned = {
        "task_pattern": str(data.get("task_pattern", ""))[:200],
        "key_steps": [str(s)[:200] for s in (data.get("key_steps") or [])][:7],
        "key_decisions": [str(d)[:200] for d in (data.get("key_decisions") or [])][:5],
        "pitfalls": [str(p)[:200] for p in (data.get("pitfalls") or [])][:4],
        "applicability": str(data.get("applicability", ""))[:200],
    }
    return cleaned


async def extract_and_store_pattern(
    *,
    objective: str,
    trajectory: dict | None,
    final_output: str,
    llm_chat_fn: Callable[[str], Awaitable[str]] | None,
    kb: Any = None,
    run_id: str = "",
) -> str | None:
    """从成功 trajectory 抽可复用 pattern, 写入 KB.

    Args:
        objective: 原始任务描述
        trajectory: save_trajectory 的 dict (含 phases/tool_calls)
        final_output: report phase 的最终输出
        llm_chat_fn: async callable, 接 prompt 返回 str. None 时跳过.
        kb: 可选 KB 实例, 不传用模块级懒加载 (auto_ingest._get_kb)
        run_id: 用于 metadata 追溯

    Returns:
        doc_id 或 None (无 LLM / 无 pattern / KB 不可用时返回 None)
    """
    if llm_chat_fn is None:
        logger.debug("trajectory pattern: no llm_chat_fn, skip")
        return None

    traj_summary = _summarize_trajectory(trajectory, final_output)
    prompt = _PATTERN_PROMPT_TEMPLATE.format(
        objective=objective[:500],
        trajectory_summary=traj_summary[:3000],
    )

    try:
        resp = await llm_chat_fn(prompt)
    except Exception:
        logger.debug("trajectory pattern LLM call failed (non-fatal)", exc_info=True)
        return None

    parsed = _parse_pattern_response(resp)
    if parsed is None:
        logger.debug("trajectory pattern: parse failed")
        return None
    if not parsed:  # 空 dict = LLM 判定无可复用内容
        logger.debug("trajectory pattern: LLM marked empty, skip")
        return None

    pattern_text = _build_pattern_text(parsed, objective)
    if not pattern_text.strip():
        return None

    # 写入 KB (复用 auto_ingest 的懒加载单例)
    if kb is None:
        try:
            from huginn.knowledge.auto_ingest import _get_kb
            kb = _get_kb()
        except Exception:
            kb = None
    if kb is None:
        logger.debug("trajectory pattern: KB unavailable, skip")
        return None

    try:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        # C3: 写入前查 task_pattern 去重. 已存在 → 视为新证据 +ε.
        task_pattern_key = parsed.get("task_pattern", "")
        existing_doc_id = _find_pattern_by_task_pattern(kb, task_pattern_key)
        if existing_doc_id:
            update_pattern_confidence(kb, existing_doc_id, success=True)
            logger.info(
                "trajectory pattern already exists: doc_id=%s, +ε (c=0.5 initial)",
                existing_doc_id,
            )
            return existing_doc_id
        result = kb.add_text(
            pattern_text,
            filename=f"pattern_{run_id or ts}",
            metadata={
                "source": "trajectory_pattern",
                "objective": objective[:200],
                "run_id": run_id,
                "task_pattern": parsed.get("task_pattern", ""),
                "confidence": "0.5",  # C3: 初始 confidence (ChromaDB metadata 是 string)
            },
        )
        doc_id = result.get("doc_id") or None
        if doc_id:
            logger.info(
                "trajectory pattern stored: doc_id=%s, task_pattern=%s, c=0.5",
                doc_id, parsed.get("task_pattern", "")[:80],
            )
        return doc_id
    except Exception:
        logger.debug("trajectory pattern KB write failed (non-fatal)", exc_info=True)
        return None


# === C3: PM Bayesian confidence 闭环 ===
# 数学: c ∈ [0,1], 写入 c_0=0.5, 复用成功 c←(c·α+1·β)/(α+β), 失败 c←(c·α+0·β)/(α+β)
# α=经验权重 (5), β=新证据权重 (1). c < c_min (0.2) 删除.
# ponytail: 直接操作 ChromaDB collection.update, 不引入新抽象层.
# 升级路径: 加 prior (按 task_pattern 类型设不同 c_0).

_C3_ALPHA = 5.0  # 经验权重
_C3_BETA = 1.0   # 新证据权重
# C3: c_min 默认 0.2, 前端 Settings 可调 (HUGINN_PM_C_MIN).
# ponytail: 模块加载时读一次 env, 运行时改 env 不生效 (需重启). 升级路径: 改成函数读.
import os as _os_c3
_C3_C_MIN = float(_os_c3.environ.get("HUGINN_PM_C_MIN", "0.2"))
_C3_C_INIT = 0.5  # 初始 confidence


def _find_pattern_by_task_pattern(kb: Any, task_pattern: str) -> str | None:
    """按 task_pattern metadata 查已存在的 pattern doc_id. 返回 doc_id 或 None.

    ponytail: 直接 collection.get 拉 source=trajectory_pattern 的全部 metadata,
    在 Python 端按 task_pattern 过滤. 不走 query (BM25+vector RRF).
    """
    if not task_pattern:
        return None
    try:
        data = kb.collection.get(
            where={"source": "trajectory_pattern"},
            include=["metadatas"],
        )
    except Exception:
        return None
    metadatas = data.get("metadatas") or []
    for meta in metadatas:
        if meta.get("task_pattern") == task_pattern:
            return meta.get("doc_id")
    return None


def update_pattern_confidence(
    kb: Any,
    doc_id: str,
    success: bool,
    *,
    alpha: float = _C3_ALPHA,
    beta: float = _C3_BETA,
    c_min: float = _C3_C_MIN,
) -> float | None:
    """Bayesian confidence 更新. 复用成功 +ε, 失败 -ε.

    返回更新后的 confidence, 或 None (doc_id 不存在 / 删除).
    c < c_min → delete_document(doc_id) 并返回 None.
    """
    if not doc_id:
        return None
    try:
        data = kb.collection.get(
            where={"doc_id": doc_id},
            include=["metadatas"],
        )
    except Exception:
        return None
    ids = data.get("ids") or []
    metadatas = data.get("metadatas") or []
    if not ids or not metadatas:
        return None
    # 取当前 confidence (默认 c_0)
    try:
        c_old = float(metadatas[0].get("confidence", _C3_C_INIT))
    except (TypeError, ValueError):
        c_old = _C3_C_INIT
    # Bayesian update: c_new = (c_old·α + success·β) / (α+β)
    c_new = (c_old * alpha + (1.0 if success else 0.0) * beta) / (alpha + beta)
    # 低于阈值 → 删除
    if c_new < c_min:
        try:
            kb.delete_document(doc_id)
        except Exception:
            logger.debug(
                "delete low-confidence pattern failed: %s", doc_id, exc_info=True,
            )
        logger.info(
            "pattern %s deleted (c=%.3f < c_min=%.2f)",
            doc_id, c_new, c_min,
        )
        return None
    # 更新所有 chunks 的 confidence metadata
    new_metadatas = []
    for meta in metadatas:
        new_meta = dict(meta)
        new_meta["confidence"] = f"{c_new:.4f}"
        new_metadatas.append(new_meta)
    try:
        kb.collection.update(ids=ids, metadatas=new_metadatas)
    except Exception:
        logger.debug(
            "update_pattern_confidence KB update failed", exc_info=True,
        )
        return None
    logger.info(
        "pattern %s confidence: %.3f → %.3f (success=%s)",
        doc_id, c_old, c_new, success,
    )
    return c_new


# === trajectory_match: line graph + VF2 子图同构 ===

def _to_line_graph(tool_seq: list[str]) -> Any:
    """tool_name 序列 → line graph (相邻关系图).

    序列 [a,b,c,d] → path graph a-b-c-d.
    VF2 子图同构: current 的 line graph 是否是 history line graph 的子图.
    ponytail: networkx.path_graph 直接用, 节点 label = tool_name.
    """
    import networkx as nx
    g = nx.path_graph(len(tool_seq))
    # 节点 label 用 tool_name, 用于 VF2 匹配
    for i, name in enumerate(tool_seq):
        g.nodes[i]["tool"] = name
    return g


def trajectory_match(
    current: list[str],
    history: list[list[str]],
    *,
    min_similarity: float = 0.5,
) -> dict | None:
    """VF2 子图同构: 当前 tool 序列是否是某历史轨迹的 prefix.

    治 spec 天花板 "trajectory KB 有写入无读取".
    _check_stuck 调本函数, 找到相似历史 → 取下一步 tool 作为建议注入 prompt.

    返回 {"history_id", "similarity", "next_step"} 或 None.
    similarity = len(current) / len(history[hid])  (prefix 匹配长度比).

    ponytail: line graph 让"序列 prefix 匹配"变成"子图同构", 复用 networkx VF2.
    不引入自定义图算法. 升级: 跨域相似度 (Jaccard on tool_name set).
    """
    import networkx as nx
    if not current or not history:
        return None

    cur_g = _to_line_graph(current)
    best: dict | None = None
    for hid, hist_seq in enumerate(history):
        if len(hist_seq) <= len(current):
            continue  # current 必须是严格 prefix (history 更长)
        hist_g = _to_line_graph(hist_seq)
        # node_match: tool_name 相等
        matcher = nx.algorithms.isomorphism.GraphMatcher(
            hist_g, cur_g,
            node_match=lambda a, b: a.get("tool") == b.get("tool"),
        )
        if matcher.subgraph_is_isomorphic():
            sim = len(current) / len(hist_seq)
            if sim >= min_similarity and (best is None or sim > best["similarity"]):
                # next_step = history 里 current 之后的第一个 tool
                next_idx = len(current)
                next_step = hist_seq[next_idx] if next_idx < len(hist_seq) else None
                best = {
                    "history_id": hid,
                    "similarity": sim,
                    "next_step": next_step,
                }
    return best


# === 自检 ===

if __name__ == "__main__":
    import asyncio

    # 1. _summarize_trajectory
    traj = {
        "phases": [{"name": "perceive"}, {"name": "hypothesize"}, {"name": "execute"}],
        "tool_calls": [
            {"tool": "vasp_run"}, {"tool": "band_structure"}, {"tool": "analysis"},
        ],
    }
    summary = _summarize_trajectory(traj, "final result text")
    assert "perceive" in summary and "hypothesize" in summary
    assert "vasp_run" in summary and "band_structure" in summary
    assert "3 total" in summary
    assert "final result text" in summary

    # 1b. trajectory None
    summary = _summarize_trajectory(None, "output")
    assert "no trajectory data" in summary and "output" in summary

    # 1c. 长 final_output 截断
    summary = _summarize_trajectory(None, "x" * 1000)
    assert len(summary) < 700  # 500 截断 + label

    # 2. _build_pattern_text
    parsed = {
        "task_pattern": "DFT band structure calculation",
        "key_steps": ["Relax structure", "SCF", "Band structure"],
        "key_decisions": ["PBE over HSE (cheaper, similar gap)"],
        "pitfalls": ["Don't forget KPOINTS path"],
        "applicability": "When user asks band gap",
    }
    text = _build_pattern_text(parsed, "compute band gap of Si")
    assert "REUSABLE PATTERN" in text
    assert "DFT band structure" in text
    assert "Relax structure" in text
    assert "PBE over HSE" in text
    assert "Don't forget KPOINTS" in text
    assert "When user asks band gap" in text

    # 2b. 空 task_pattern → 空文本
    text = _build_pattern_text({"task_pattern": ""}, "obj")
    assert text == ""

    # 3. _parse_pattern_response
    resp = '{"task_pattern": "X", "key_steps": ["a", "b"], "key_decisions": ["c"], "pitfalls": ["d"], "applicability": "e"}'
    p = _parse_pattern_response(resp)
    assert p is not None
    assert p["task_pattern"] == "X"
    assert p["key_steps"] == ["a", "b"]
    assert p["key_decisions"] == ["c"]

    # 3b. markdown fence
    p = _parse_pattern_response('```json\n{"task_pattern": "Y"}\n```')
    assert p is not None and p["task_pattern"] == "Y"

    # 3c. 空 pattern (LLM 判定无可复用) → 空 dict
    p = _parse_pattern_response('{"task_pattern": ""}')
    assert p == {}

    # 3d. 非 JSON → None
    assert _parse_pattern_response("not json") is None
    assert _parse_pattern_response("") is None
    assert _parse_pattern_response(None) is None

    # 3e. 字段长度截断
    long_str = "x" * 500
    p = _parse_pattern_response(
        f'{{"task_pattern": "{long_str}", "key_steps": ["{long_str}"]}}')
    assert len(p["task_pattern"]) <= 200
    assert len(p["key_steps"][0]) <= 200

    # 3f. 字段数量截断
    many_steps = [f"s{i}" for i in range(20)]
    p = _parse_pattern_response(
        json.dumps({"task_pattern": "X", "key_steps": many_steps}))
    assert len(p["key_steps"]) <= 7

    # 4. extract_and_store_pattern — 无 LLM 时跳过
    async def _run_no_llm():
        doc_id = await extract_and_store_pattern(
            objective="test", trajectory=traj, final_output="out",
            llm_chat_fn=None)
        assert doc_id is None, "no llm → None"

    asyncio.run(_run_no_llm())

    # 5. extract_and_store_pattern — LLM 给空 pattern 时跳过
    async def _empty_llm(prompt: str) -> str:
        return '{"task_pattern": ""}'

    async def _run_empty_pattern():
        doc_id = await extract_and_store_pattern(
            objective="test", trajectory=traj, final_output="out",
            llm_chat_fn=_empty_llm, kb=object())  # kb 不被调用
        assert doc_id is None, "empty pattern → None"

    asyncio.run(_run_empty_pattern())

    # 6. extract_and_store_pattern — mock LLM + mock KB
    async def _good_llm(prompt: str) -> str:
        return ('{"task_pattern": "X", "key_steps": ["a"], "key_decisions": [], '
                '"pitfalls": [], "applicability": "when X"}')

    class _MockKB:
        def __init__(self):
            self.calls = []
        def add_text(self, text, filename="", metadata=None):
            self.calls.append({"text": text, "filename": filename, "metadata": metadata})
            return {"doc_id": "doc_123", "chunks": 2}

    async def _run_good():
        mock_kb = _MockKB()
        doc_id = await extract_and_store_pattern(
            objective="compute band gap", trajectory=traj,
            final_output="gap=1.1eV", llm_chat_fn=_good_llm, kb=mock_kb,
            run_id="r1")
        assert doc_id == "doc_123"
        assert len(mock_kb.calls) == 1
        call = mock_kb.calls[0]
        assert "REUSABLE PATTERN" in call["text"]
        assert call["metadata"]["source"] == "trajectory_pattern"
        assert call["metadata"]["run_id"] == "r1"
        assert call["metadata"]["task_pattern"] == "X"

    asyncio.run(_run_good())

    # 7. extract_and_store_pattern — LLM 抛异常时返回 None
    async def _raise_llm(prompt: str) -> str:
        raise RuntimeError("LLM offline")

    async def _run_llm_raise():
        mock_kb = _MockKB()
        doc_id = await extract_and_store_pattern(
            objective="test", trajectory=traj, final_output="out",
            llm_chat_fn=_raise_llm, kb=mock_kb)
        assert doc_id is None
        assert len(mock_kb.calls) == 0, "LLM 失败不应写 KB"

    asyncio.run(_run_llm_raise())

    # 8. extract_and_store_pattern — KB 不可用时返回 None
    async def _run_no_kb():
        # kb=None 且 auto_ingest._get_kb 也拿不到 (无 chromadb 环境)
        # 直接传一个 mock None
        doc_id = await extract_and_store_pattern(
            objective="test", trajectory=traj, final_output="out",
            llm_chat_fn=_good_llm, kb=None)
        # 没有 chromadb 环境时 _get_kb 返回 None, 结果 None
        # 如果有 chromadb, 会真的写入 — 这里不强断言, 只看是否崩
        assert doc_id is None or isinstance(doc_id, str)

    asyncio.run(_run_no_kb())

    # 9. trajectory_match — line graph + VF2 子图同构
    # 历史 [[a,b,c,d], [a,b,e,f]], 当前 [a,b,c] → 匹配 hid=0, sim=0.75
    history = [["a", "b", "c", "d"], ["a", "b", "e", "f"]]
    current = ["a", "b", "c"]
    match = trajectory_match(current, history)
    assert match is not None, f"应匹配, got {match}"
    assert match["history_id"] == 0, f"应匹配 hid=0, got {match['history_id']}"
    assert abs(match["similarity"] - 0.75) < 1e-9, f"sim 应 0.75, got {match['similarity']}"
    assert match["next_step"] == "d", f"next_step 应 d, got {match['next_step']}"
    print(f"[ok] trajectory_match([a,b,c], history) → hid={match['history_id']}, sim={match['similarity']}, next={match['next_step']}")

    # 9b. 当前 [a,b,e] → 匹配 hid=1
    current2 = ["a", "b", "e"]
    match2 = trajectory_match(current2, history)
    assert match2 is not None and match2["history_id"] == 1, f"应 hid=1, got {match2}"
    assert match2["next_step"] == "f"
    print(f"[ok] trajectory_match([a,b,e], history) → hid=1, next=f")

    # 9c. 当前 [x,y,z] 无匹配 → None
    match3 = trajectory_match(["x", "y", "z"], history)
    assert match3 is None, f"无匹配应 None, got {match3}"
    print(f"[ok] trajectory_match([x,y,z], history) → None")

    # 9d. 当前比历史长 → None (子图同构要求 current ⊆ history)
    match4 = trajectory_match(["a", "b", "c", "d", "e"], history)
    assert match4 is None, f"current 比历史长应 None, got {match4}"
    print(f"[ok] current 比历史长 → None")

    # === C3: PM Bayesian confidence 闭环 (用 mock KB) ===
    class _MockKB:
        """Mock KB 只实现 C3 需要的接口: collection.get/update, delete_document."""
        def __init__(self):
            # 存储: {id: {metadata: dict, document: str}}
            self._store: dict[str, dict] = {}
            self.collection = self  # kb.collection == kb (self)

        def add_text(self, text, filename="", metadata=None):
            import uuid
            doc_id = uuid.uuid4().hex[:12]
            meta = dict(metadata or {})
            meta["doc_id"] = doc_id
            self._store[doc_id] = {"metadata": meta, "document": text}
            return {"doc_id": doc_id, "chunks": 1}

        def get(self, where=None, include=None):
            # 按 where 过滤 (只支持 source / doc_id 单字段)
            src = where.get("source") if where else None
            did = where.get("doc_id") if where else None
            ids, metas = [], []
            for k, v in self._store.items():
                m = v["metadata"]
                if src and m.get("source") != src:
                    continue
                if did and m.get("doc_id") != did:
                    continue
                ids.append(k)
                metas.append(dict(m))
            return {"ids": ids, "metadatas": metas}

        def update(self, ids=None, metadatas=None):
            for i, mid in enumerate(ids or []):
                if mid in self._store and i < len(metadatas or []):
                    self._store[mid]["metadata"] = dict(metadatas[i])

        def delete_document(self, doc_id):
            # 删除所有 chunks (mock 简化: doc_id 是 metadata.doc_id, 按 metadata 匹配)
            keys_to_del = [
                k for k, v in self._store.items()
                if v["metadata"].get("doc_id") == doc_id
            ]
            for k in keys_to_del:
                del self._store[k]
            return True

    mock_kb = _MockKB()
    # case C3-A: 写入 → confidence=0.5
    doc_id_a = mock_kb.add_text(
        "REUSABLE PATTERN (from successful task: GaN band gap)\nTask type: dft_band_gap",
        filename="pattern_test1",
        metadata={
            "source": "trajectory_pattern",
            "task_pattern": "dft_band_gap",
            "confidence": "0.5",
        },
        )["doc_id"]
    # case C3-B: 复用成功 → +ε
    c1 = update_pattern_confidence(mock_kb, doc_id_a, success=True)
    assert c1 is not None, "success update should return new c"
    assert c1 > 0.5, f"success should increase c, got {c1}"
    print(f"[ok] C3-B success +ε: 0.5 → {c1:.4f}")

    # case C3-C: 复用失败 → -ε
    c2 = update_pattern_confidence(mock_kb, doc_id_a, success=False)
    assert c2 is not None, "fail update should return new c"
    assert c2 < c1, f"fail should decrease c, got {c2} (prev {c1})"
    print(f"[ok] C3-C fail -ε: {c1:.4f} → {c2:.4f}")

    # case C3-D: 多次失败 → c < c_min 删除
    c_prev = c2
    deleted = False
    for _ in range(20):  # 足够多次失败让 c 跌破 0.2
        c_new = update_pattern_confidence(mock_kb, doc_id_a, success=False)
        if c_new is None:
            deleted = True
            break
        c_prev = c_new
    assert deleted, f"应被删除 (c 跌破 0.2), 最后 c={c_prev:.4f}"
    print(f"[ok] C3-D multiple fails → deleted (c < c_min=0.2)")

    # case C3-E: _find_pattern_by_task_pattern 去重
    mock_kb2 = _MockKB()
    mock_kb2.add_text(
        "PATTERN 1",
        metadata={"source": "trajectory_pattern", "task_pattern": "type_X", "confidence": "0.5"},
    )
    found = _find_pattern_by_task_pattern(mock_kb2, "type_X")
    assert found, "应能找到 task_pattern=type_X"
    not_found = _find_pattern_by_task_pattern(mock_kb2, "type_Y")
    assert not_found is None, "type_Y 不存在"
    print(f"[ok] C3-E _find_pattern_by_task_pattern dedup OK")

    print("trajectory_pattern selfcheck All passed")
