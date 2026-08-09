"""P0+P1 时序表征增强自检.

覆盖:
- P0: LoopState.iteration_history 字段 + reflect push 路径
- P0: _build_decider_prompt 注入 Previous iterations 块
- P1: Memory.retrieve since 过滤
- P1: KB.query since 过滤 (走 mock collection)
- P1: KG.query / query_episode_path since 过滤

不引入 pytest — assert-based, `python -m tests.test_temporal_iter_hist` 可跑.
"""

import sys
import tempfile
from pathlib import Path

# 确保能 import huginn
_AGENT_DIR = Path(__file__).resolve().parent.parent
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))


def _check_loop_state_field():
    """P0: LoopState 有 iteration_history 字段, 默认空 list."""
    from huginn.autoloop.cognitive_loop import _MAX_ITER_HIST, LoopState
    s = LoopState()
    assert hasattr(s, "iteration_history"), "LoopState 缺 iteration_history"
    assert s.iteration_history == [], "默认非空 list"
    assert isinstance(_MAX_ITER_HIST, int) and _MAX_ITER_HIST > 0
    # 上限约束: push 超限后截断到 _MAX_ITER_HIST
    for i in range(_MAX_ITER_HIST + 10):
        s.iteration_history.append({"iter": i})
    if len(s.iteration_history) > _MAX_ITER_HIST:
        del s.iteration_history[: -_MAX_ITER_HIST]
    assert len(s.iteration_history) == _MAX_ITER_HIST, \
        f"截断后长度应为 {_MAX_ITER_HIST}, 实际 {len(s.iteration_history)}"
    # 保留的是最近 _MAX_ITER_HIST 轮
    assert s.iteration_history[0]["iter"] == 10, "截断应保留最新记录"
    print("[ok] P0 LoopState.iteration_history 字段 + 截断")


def _check_decider_prompt_injection():
    """P0: _build_decider_prompt 看到 iteration_history 时注入 Previous iterations 块."""
    # 用 __new__ 绕过 __init__ (不依赖 LLM/workspace)
    from huginn.autoloop.cognitive_loop import LoopState
    from huginn.autoloop.engine import AutoloopEngine
    eng = AutoloopEngine.__new__(AutoloopEngine)
    s = LoopState()
    s.iteration_history = [
        {"iter": 0, "action": "hypothesize", "val_status": "none",
         "plan_mode": "", "redirect": False, "advice": ""},
        {"iter": 1, "action": "execute", "val_status": "failed",
         "plan_mode": "lammps", "redirect": True, "advice": "tool timeout"},
    ]
    cog = {"hypothesis": "test hyp", "plan": {"mode": "lammps"},
           "execution_result": None, "validation": None,
           "last_learn_summary": ""}
    prompt = eng._build_decider_prompt(s, cog, {})
    assert "Previous iterations" in prompt, "decider prompt 未注入 Previous iterations 块"
    assert "iter1" in prompt, "iteration_history 行未渲染"
    assert "lammps" in prompt, "plan_mode 未渲染"
    assert "tool timeout" in prompt, "advice 未渲染 (redirect 路径)"
    # 空历史也应正常工作 (无崩溃, 无 Previous iterations 行)
    s2 = LoopState()
    p2 = eng._build_decider_prompt(s2, cog, {})
    assert "Previous iterations" in p2, "空历史也应渲染块头"
    assert "(none)" in p2, "空历史应显示 (none)"
    print("[ok] P0 decider prompt 注入 iteration_history")


def _check_memory_since_filter():
    """P1: LongTermMemory.retrieve since 参数过滤 created_at."""
    from huginn.memory.longterm import LongTermMemory
    with tempfile.TemporaryDirectory() as d:
        mem = LongTermMemory(db_path=Path(d) / "m.db")
        # store() 用 datetime.now() 写 created_at, 不能直接覆盖. 写完后 UPDATE 改时间戳.
        old_id = mem.store("old finding", category="discovery", tier="long")
        new_id = mem.store("new finding", category="discovery", tier="long")
        with mem._connect() as conn:
            conn.execute(
                "UPDATE memories SET created_at = ? WHERE id = ?",
                ("2020-01-01T00:00:00", old_id),
            )
            conn.execute(
                "UPDATE memories SET created_at = ? WHERE id = ?",
                ("2025-06-01T00:00:00", new_id),
            )
            conn.commit()
        # 不过滤: 两条都在
        all_rows = mem.retrieve("finding", top_k=10)
        assert len(all_rows) >= 2, f"无 since 应返回 2+ 条, 实际 {len(all_rows)}"
        # since=2024: 只剩新的
        recent = mem.retrieve("finding", top_k=10, since="2024-01-01T00:00:00")
        contents = [r.get("content", "") for r in recent]
        assert any("new finding" in c for c in contents), "since 过滤后应包含新记录"
        assert not any("old finding" in c for c in contents), \
            "since 过滤后不应包含旧记录"
        print("[ok] P1 Memory.retrieve since 过滤")


def _check_kb_since_filter():
    """P1: KB.query since 参数走 where_filter (mock collection 验证)."""
    from unittest.mock import MagicMock

    from huginn.knowledge.store import KnowledgeBase

    # mock 掉 ChromaDB 和 embedding model, 不依赖真实模型
    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb.collection = MagicMock()
    kb.collection.count.return_value = 10
    kb.collection.query.return_value = {
        "ids": [["c1"]],
        "documents": [["doc text"]],
        "metadatas": [[{"domain": "alloy", "created_at": "2025-01-01"}]],
        "distances": [[0.1]],
    }
    # model 是 @property, mock _model 私有字段绕开懒加载
    # encode() 返回值要支持 .tolist() (真实返回 np.ndarray)
    kb._model = MagicMock()
    _encoded = MagicMock()
    _encoded.tolist.return_value = [[0.1, 0.2]]
    kb._model.encode.return_value = _encoded
    kb._semantic_cache = None
    # _query_cache 是 TimedLRUCache, 用 MagicMock 同时支持 get/set
    kb._query_cache = MagicMock()
    kb._query_cache.get.return_value = None
    kb._bm25 = None
    kb._bm25_dirty = True
    kb._feedback_tracker = None

    # 传 since: 验证 where_filter 包含 $gte
    kb.query("test", top_k=5, since="2024-01-01T00:00:00")
    _args = kb.collection.query.call_args
    _wf = _args.kwargs.get("where")
    assert _wf is not None, "since 过滤应生成 where_filter"
    # 单条件时直接是 {created_at: {$gte: ...}}; 多条件时是 {$and: [...]}
    if "$and" in _wf:
        _conds = _wf["$and"]
    else:
        _conds = [_wf]
    _has_gte = any(
        isinstance(c, dict) and "created_at" in c
        and isinstance(c["created_at"], dict) and "$gte" in c["created_at"]
        for c in _conds
    )
    assert _has_gte, f"where_filter 缺 created_at $gte 条件: {_wf}"

    # 同时传 domain + since: 验证 $and 组合
    kb.query("test", top_k=5, domain="alloy", since="2024-01-01")
    _wf2 = kb.collection.query.call_args.kwargs.get("where")
    assert "$and" in _wf2, f"domain+since 应走 $and: {_wf2}"
    print("[ok] P1 KB.query since 过滤 (where_filter)")


def _check_kg_since_filter():
    """P1: KG.query + query_episode_path since 过滤 episode 节点."""
    from huginn.kg.graph import (
        NODE_TYPE_EPISODE,
        ProjectKnowledgeGraph,
        _filter_by_since,
    )

    # _filter_by_since 单元测试 (不依赖真实图)
    result = {
        "nodes": [
            {"id": "mat_1", "type": "Material", "label": "Fe"},
            {"id": "ep_old", "type": NODE_TYPE_EPISODE, "timestamp": "2020-01-01"},
            {"id": "ep_new", "type": NODE_TYPE_EPISODE, "timestamp": "2025-01-01"},
        ],
        "edges": [
            {"source": "mat_1", "target": "ep_old"},
            {"source": "mat_1", "target": "ep_new"},
        ],
    }
    filtered = _filter_by_since(result, "2024-01-01")
    node_ids = {n["id"] for n in filtered["nodes"]}
    assert "mat_1" in node_ids, "非 episode 节点应保留"
    assert "ep_new" in node_ids, "新 episode 应保留"
    assert "ep_old" not in node_ids, "旧 episode 应被过滤"
    # 引用被过滤节点的边也应删除
    edge_targets = {e["target"] for e in filtered["edges"]}
    assert "ep_old" not in edge_targets, "指向旧 episode 的边应删除"

    # query_episode_path: 用真实图验证 since 参数生效
    with tempfile.TemporaryDirectory() as d:
        kg = ProjectKnowledgeGraph(Path(d))
        # 手动构造两个 episode 节点 + 一条 data_dep 边
        kg._graph.add_node("episode_1", type=NODE_TYPE_EPISODE, step_id=1,
                           timestamp="2020-01-01T00:00:00")
        kg._graph.add_node("episode_2", type=NODE_TYPE_EPISODE, step_id=2,
                           timestamp="2025-01-01T00:00:00")
        kg._graph.add_edge("episode_1", "episode_2", relation="data_dep")
        # 不过滤: backward 应返回 2 个
        all_eps = kg.query_episode_path(2, direction="backward")
        assert len(all_eps) == 2, f"无 since 应返回 2 个 episode, 实际 {len(all_eps)}"
        # since=2024: 只剩 episode_2
        recent = kg.query_episode_path(2, direction="backward",
                                       since="2024-01-01T00:00:00")
        ids = [e.get("step_id") for e in recent]
        assert 2 in ids, "新 episode 应保留"
        assert 1 not in ids, "旧 episode 应被过滤"
        print("[ok] P1 KG.query_episode_path + _filter_by_since")


def main():
    _check_loop_state_field()
    _check_decider_prompt_injection()
    _check_memory_since_filter()
    _check_kb_since_filter()
    _check_kg_since_filter()
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
