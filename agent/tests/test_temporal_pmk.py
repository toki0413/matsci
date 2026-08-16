"""时序表征 + PMK 循环结合自检.

覆盖:
- 结合1: build_pmk_state 的 since/mem_mgr/timeseries_ctx 参数
- 结合2: check_pause_decision 的 iteration_history 重复冲突升级
- 结合3: _check_pmk_consistency 识别 timeseries 第四路 (PMKT)

不引入 pytest — assert-based, `python -m tests.test_temporal_pmk` 可跑.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

_AGENT_DIR = Path(__file__).resolve().parent.parent
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))


def _check_pmk_since_kb():
    """结合1: build_pmk_state 传 since 时, kb 检索收到 since 参数.

    代码走 query_with_dedup 优先 (桥 RAG), 无该方法时回退 kb.query.
    """
    from huginn.autoloop.cognitive_guard import build_pmk_state

    kb = MagicMock()
    kb.query_with_dedup.return_value = [{"content": "kb hit"}]
    step_eval = MagicMock()
    step_eval.attempted = "Fe diffusion"
    step_eval.pmk_feedback = "memory: some mem"

    # 不传 since — query_with_dedup 不应有 since
    build_pmk_state(None, step_eval, kb)
    _args = kb.query_with_dedup.call_args
    assert _args.kwargs.get("since") is None, \
        f"无 since 时检索不应传 since, 实际 {_args.kwargs}"

    # 传 since — query_with_dedup 应收到
    build_pmk_state(None, step_eval, kb, since="2025-01-01T00:00:00")
    _args2 = kb.query_with_dedup.call_args
    assert _args2.kwargs.get("since") == "2025-01-01T00:00:00", \
        f"有 since 时检索应传 since, 实际 {_args2.kwargs}"

    print("[ok] 结合1: build_pmk_state → kb query(since=...)")


def _check_pmk_since_memory():
    """结合1: build_pmk_state 传 mem_mgr + since 时, recall_for_prompt 收到 since."""
    from huginn.autoloop.cognitive_guard import build_pmk_state

    mem_mgr = MagicMock()
    mem_mgr.recall_for_prompt.return_value = "recent mem text"
    kb = MagicMock()
    kb.query.return_value = []
    step_eval = MagicMock()
    step_eval.attempted = "test"
    step_eval.pmk_feedback = "memory: base mem"

    # 不传 mem_mgr/since — M 路只有 pmk_feedback
    _r1 = build_pmk_state(None, step_eval, kb)
    assert _r1["memory"] == "base mem", "无 since 时 M 路应只有 pmk_feedback"

    # 传 mem_mgr + since — M 路应包含 recall 结果
    _r2 = build_pmk_state(None, step_eval, kb, since="2025-01-01", mem_mgr=mem_mgr)
    assert "recent mem text" in _r2["memory"], "M 路应包含 recall_for_prompt 结果"
    assert "base mem" in _r2["memory"], "M 路应保留原 pmk_feedback"
    _call = mem_mgr.recall_for_prompt.call_args
    assert _call.kwargs.get("since") == "2025-01-01", "recall_for_prompt 应收到 since"

    print("[ok] 结合1: build_pmk_state → recall_for_prompt(since=...)")


def _check_pmk_timeseries_route():
    """结合3: build_pmk_state 传 timeseries_ctx 时, 结果含 timeseries 第四路."""
    from huginn.autoloop.cognitive_guard import build_pmk_state

    step_eval = MagicMock()
    step_eval.attempted = "Fe"
    step_eval.pmk_feedback = "memory: test"

    # 不传 timeseries_ctx — 无 timeseries key
    _r1 = build_pmk_state(None, step_eval, None)
    assert "timeseries" not in _r1, "无 timeseries_ctx 时不应有 timeseries key"

    # 传 timeseries_ctx — 有 timeseries key
    _ts_ctx = "VACF (lammps): 3 pts, trend=decaying. meaning: velocity autocorrelation"
    _r2 = build_pmk_state(None, step_eval, None, timeseries_ctx=_ts_ctx)
    assert "timeseries" in _r2, "有 timeseries_ctx 时应有 timeseries key"
    assert "decaying" in _r2["timeseries"], "timeseries 内容应保留"

    print("[ok] 结合3: build_pmk_state → timeseries 第四路 (PMKT)")


def _check_pmk_consistency_timeseries():
    """结合3: _check_pmk_consistency 识别 timeseries 路, 跟 persona 冲突."""
    from huginn.runtime.task_lifecycle import _check_pmk_consistency

    # VACF decaying (扩散) 但 persona 说束缚态 → 冲突
    _pmk = {
        "persona": "I recommend 束缚态 stable bound state for this system",
        "timeseries": "VACF (lammps): 3 pts, trend=decaying. meaning: velocity autocorrelation",
    }
    _inconsistent, _reason = _check_pmk_consistency(_pmk)
    assert _inconsistent, "VACF decaying + persona 束缚态应触发冲突"
    assert "timeseries" in _reason or "persona" in _reason, \
        f"冲突 reason 应提到 timeseries 或 persona, 实际: {_reason}"

    # VACF flat (平衡) + persona 说平衡 → 不冲突
    _pmk2 = {
        "persona": "I recommend 平衡 equilibrium state",
        "timeseries": "Temp: 2 pts, trend=flat. meaning: temperature",
    }
    _inc2, _ = _check_pmk_consistency(_pmk2)
    # 都 recommend 平衡 → 不冲突
    assert not _inc2, "VACF flat + persona 平衡不应冲突"

    # 无 timeseries — 原有三路行为不变
    _pmk3 = {
        "persona": "I recommend approach A",
        "memory": "I oppose approach A",
    }
    _inc3, _reason3 = _check_pmk_consistency(_pmk3)
    assert _inc3, "原有三路冲突应正常工作"

    print("[ok] 结合3: _check_pmk_consistency 识别 timeseries 路")


def _check_pause_iteration_history():
    """结合2: check_pause_decision 传 iteration_history, 重复 redirect 升级 reason."""
    from huginn.autoloop.cognitive_guard import check_pause_decision

    # mock should_pause_for_decision 返回 pause=True
    from huginn.runtime import task_lifecycle as _tl
    _orig = _tl.should_pause_for_decision
    _tl.should_pause_for_decision = lambda *a, **k: (True, "PMK conflict", [])

    try:
        # 不传 iteration_history — reason 不升级
        _pause, _reason, _ = check_pause_decision([], [], None, None, {})
        assert _pause
        assert _reason == "PMK conflict", f"无 iter_hist 时 reason 不应升级: {_reason}"

        # 传 iteration_history, 3/5 redirect — reason 升级
        _hist = [
            {"iter": i, "redirect": True} for i in range(3)
        ] + [{"iter": i, "redirect": False} for i in range(2)]
        _pause2, _reason2, _ = check_pause_decision(
            [], [], None, None, {}, iteration_history=_hist,
        )
        assert _pause2
        assert "repeated PMK conflict" in _reason2, \
            f"3/5 redirect 应升级 reason: {_reason2}"
        assert "3/5" in _reason2, "reason 应包含 redirect 计数"

        # 传 iteration_history, 1/5 redirect — 不升级
        _hist2 = [{"iter": 0, "redirect": True}] + [
            {"iter": i, "redirect": False} for i in range(1, 5)
        ]
        _pause3, _reason3, _ = check_pause_decision(
            [], [], None, None, {}, iteration_history=_hist2,
        )
        assert _reason3 == "PMK conflict", f"1/5 redirect 不应升级: {_reason3}"
    finally:
        _tl.should_pause_for_decision = _orig

    print("[ok] 结合2: check_pause_decision iteration_history 重复冲突升级")


def _qualia_probe_self_reference():
    """感质探针: self_eval 自指路与 persona/memory/kb 世界路对同一 subject 冲突 →
    自指固定点. 这是"同一直由度既当世界事实又当系统自我状态, 不可分离"的工程探针."""
    from huginn.runtime.task_lifecycle import _check_pmk_consistency

    # persona (世界路, 赞成 GNN 方法) vs self_eval (自指路, 自我报告自己用不好 GNN)
    # 同一 subject "gnn" 被 persona 当客观方法论主张、被 self_eval 当自身状态 → 冲突
    _pmk_q = {
        "persona": "I recommend GNN approach for this task",
        "self_eval": "self-report: my GNN attempt failed, I now oppose continuing GNN",
    }
    _inc, _reason = _check_pmk_consistency(_pmk_q)
    assert _inc, "world persona + self_eval 对同一 subject 冲突应触发"
    assert "qualia-probe" in _reason, \
        f"自指冲突 reason 应带 qualia-probe 标记, 实际: {_reason}"

    # pure 世界冲突 (persona vs memory) — 不涉及自指路 → 无 qualia 标记
    _pmk_w = {
        "persona": "I recommend approach A",
        "memory": "I oppose approach A",
    }
    _inc2, _reason2 = _check_pmk_consistency(_pmk_w)
    assert _inc2, "pure 世界路冲突应触发"
    assert "qualia-probe" not in _reason2, \
        f"纯世界冲突不应带 qualia 标记, 实际: {_reason2}"

    print("[ok] 感质探针: self_eval 自指路冲突 → qualia-probe 自指固定点")


def main():
    _check_pmk_since_kb()
    _check_pmk_since_memory()
    _check_pmk_timeseries_route()
    _check_pmk_consistency_timeseries()
    _check_pause_iteration_history()
    _qualia_probe_self_reference()
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
