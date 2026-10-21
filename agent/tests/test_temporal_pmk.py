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
    """三轴: memory/kb/timeseries 属认识论（advisory），persona/action 属规范（block）."""
    from huginn.runtime.task_lifecycle import _check_pmk_axes, _check_pmk_consistency

    # 认识论: memory 说束缚态 vs timeseries decaying → advisory, 不阻断
    _pmk = {
        "memory": "I recommend 束缚态 stable bound state for this system",
        "timeseries": "VACF (lammps): 3 pts, trend=decaying. meaning: velocity autocorrelation",
    }
    _blocked, _reason = _check_pmk_consistency(_pmk)
    assert not _blocked, "认识论冲突只应 advisory, 不应阻断"
    assert "EPISTEMIC" in _reason, _reason
    _axes = _check_pmk_axes(_pmk)
    assert _axes.advisory and not _axes.blocked

    # 认识论: flat + 平衡 → 不冲突
    _pmk2 = {
        "memory": "I recommend 平衡 equilibrium state",
        "timeseries": "Temp: 2 pts, trend=flat. meaning: temperature",
    }
    _blocked2, _ = _check_pmk_consistency(_pmk2)
    assert not _blocked2, "memory 平衡 + timeseries flat 不应冲突"

    # 认识论: memory vs kb 冲突 → advisory
    _pmk3 = {
        "memory": "I recommend approach A",
        "kb": "I oppose approach A",
    }
    _blocked3, _reason3 = _check_pmk_consistency(_pmk3)
    assert not _blocked3, "memory/kb 冲突属认识论, 不应阻断"
    assert "EPISTEMIC" in _reason3, _reason3

    # 规范: persona 反对 action 推荐 → block
    _pmk4 = {
        "persona": "I oppose approach A",
        "action": "I recommend approach A",
    }
    _blocked4, _reason4 = _check_pmk_consistency(_pmk4)
    assert _blocked4, "persona/action 冲突应阻断"
    assert "NORMATIVE" in _reason4, _reason4

    print("[ok] 三轴: 认识论 advisory / 规范 block")


def _check_pause_iteration_history():
    """结合2: check_pause_decision 传 iteration_history, 重复 redirect 升级 reason."""
    from huginn.autoloop.cognitive_guard import check_pause_decision

    # mock should_pause_for_decision 返回 pause=True
    from huginn.runtime import task_lifecycle as _tl
    _orig = _tl.should_pause_for_decision
    _tl.should_pause_for_decision = lambda *a, **k: (True, "decision conflict", [])

    try:
        # 不传 iteration_history — reason 不升级
        _pause, _reason, _ = check_pause_decision([], [], None, None, {})
        assert _pause
        assert _reason == "decision conflict", f"无 iter_hist 时 reason 不应升级: {_reason}"

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
        assert _reason3 == "decision conflict", f"1/5 redirect 不应升级: {_reason3}"
    finally:
        _tl.should_pause_for_decision = _orig

    print("[ok] 结合2: check_pause_decision iteration_history 重复冲突升级")


def _qualia_probe_self_reference():
    """感质探针: persona（身份/规范）与 self_eval（自指报告）对同一 subject 冲突 →
    自指固定点；与 persona/action 的规范冲突区分开（后者不带 qualia 标记）."""
    from huginn.runtime.task_lifecycle import _check_pmk_consistency

    # persona (身份: 赞成 GNN 方法) vs self_eval (自指路: 自我报告自己用不好 GNN)
    # 同一 subject "gnn" 被 persona 当身份主张、被 self_eval 当自身状态 → 自指冲突
    _pmk_q = {
        "persona": "I recommend GNN approach for this task",
        "self_eval": "self-report: my GNN attempt failed, I now oppose continuing GNN",
    }
    _inc, _reason = _check_pmk_consistency(_pmk_q)
    assert _inc, "persona + self_eval 对同一 subject 冲突应触发"
    assert "qualia-probe" in _reason, \
        f"自指冲突 reason 应带 qualia-probe 标记, 实际: {_reason}"

    # 纯规范冲突 (persona vs action) — 不涉及自指路 → 无 qualia 标记
    _pmk_w = {
        "persona": "I recommend approach A",
        "action": "I oppose approach A",
    }
    _inc2, _reason2 = _check_pmk_consistency(_pmk_w)
    assert _inc2, "persona/action 规范冲突应触发"
    assert "NORMATIVE" in _reason2, _reason2
    assert "qualia-probe" not in _reason2, \
        f"纯规范冲突不应带 qualia 标记, 实际: {_reason2}"

    print("[ok] 感质探针: self_eval 自指路冲突 → qualia-probe 自指固定点")


def test_oppose_oppose_is_alignment_not_conflict():
    """回归: 两路都反对同一 subject 是一致，不是冲突（旧实现误报）."""
    from huginn.runtime.task_lifecycle import _check_pmk_consistency

    _pmk = {
        "persona": "I oppose approach A",
        "memory": "I oppose approach A",
    }
    _inc, _reason = _check_pmk_consistency(_pmk)
    assert not _inc, f"oppose+oppose 不应判冲突, 实际: {_reason}"
    assert _reason == "", _reason


def test_pmk_shadow_does_not_pause(monkeypatch):
    """PMK 默认 shadow: 评估/注入但不在冲突时暂停；enforce 才阻断."""
    from huginn.autoloop.cognitive_guard import check_pause_decision
    from huginn.runtime import task_lifecycle as _tl

    monkeypatch.setattr(
        _tl,
        "should_pause_for_decision",
        lambda *a, **k: (True, "PMK 一致性障碍: persona vs memory", []),
    )
    pmk_state = {"persona": "推荐 A", "memory": "反对 A"}

    monkeypatch.setenv("HUGINN_PMK_MODE", "shadow")
    _pause, _reason, _ = check_pause_decision([], [], None, None, pmk_state)
    assert not _pause, "shadow 模式不应因 PMK 冲突暂停"
    assert "advisory" in _reason, _reason

    monkeypatch.setenv("HUGINN_PMK_MODE", "enforce")
    _pause2, _reason2, _ = check_pause_decision([], [], None, None, pmk_state)
    assert _pause2, "enforce 模式应暂停"
    assert "PMK" in _reason2, _reason2


def test_pmk_llm_checker_advisory_and_cache():
    """LLM 只判认识论且只给 advisory；规则阻断优先，不调 LLM；同文本缓存."""
    import asyncio

    from huginn.runtime.task_lifecycle import check_pmk_consistency_llm

    calls = {"n": 0}

    async def _fake_llm(prompt: str) -> str:
        calls["n"] += 1
        return '{"conflict": true, "reason": "same subject, different wording"}'

    _pmk = {
        "memory": "use GNN for this task",
        "kb": "avoid graph neural networks",
    }
    _blocked, _reason = asyncio.run(check_pmk_consistency_llm(_pmk, _fake_llm))
    assert not _blocked, "认识论 LLM 冲突只应 advisory"
    assert "LLM advisory" in _reason, _reason

    _blocked2, _ = asyncio.run(check_pmk_consistency_llm(_pmk, _fake_llm))
    assert not _blocked2
    assert calls["n"] == 1, "同文本应命中缓存"

    # 规则轴已阻断 → 不调 LLM
    calls["n"] = 0
    _blocked3, _reason3 = asyncio.run(
        check_pmk_consistency_llm(
            {"persona": "I oppose approach A", "action": "I recommend approach A"},
            _fake_llm,
        )
    )
    assert _blocked3, _reason3
    assert calls["n"] == 0, "规则阻断优先，不应调 LLM"


def test_pmk_llm_advisory_async_reflect_path(monkeypatch):
    """异步 reflect 路径: LLM 语义判定只给 advisory，并写入 episodic."""
    import asyncio

    from huginn.autoloop.engine_reflect import EngineReflect

    monkeypatch.setenv("HUGINN_PMK_INJECT", "1")
    monkeypatch.setenv("HUGINN_PMK_MODE", "enforce")

    class _Engine:
        _last_pmk_state = {
            "memory": "use GNN for this task",
            "kb": "avoid graph neural networks",
        }
        _speculator_hint = ""

        def __init__(self):
            self.written = []

        async def _llm_chat(self, prompt):
            return '{"conflict": true, "reason": "same subject, different wording"}'

        def _write_pmk_conflict_to_episodic(self, state, reason):
            self.written.append(reason)

    engine = _Engine()
    asyncio.run(EngineReflect(engine)._pmk_llm_advisory())
    assert "LLM advisory" in engine._speculator_hint
    assert engine.written, "advisory 应写入 episodic"


def test_pmk_llm_advisory_degrades_on_timeout(monkeypatch):
    """超时/LLM 失败只损失 advisory，不抛出、不阻断主循环."""
    import asyncio

    from huginn.autoloop import engine_reflect

    monkeypatch.setenv("HUGINN_PMK_INJECT", "1")
    monkeypatch.setenv("HUGINN_PMK_MODE", "enforce")
    monkeypatch.setattr(engine_reflect, "_PMK_LLM_TIMEOUT_S", 0.05)

    class _Engine:
        _last_pmk_state = {"memory": "use GNN", "kb": "avoid GNN"}
        _speculator_hint = ""

        async def _llm_chat(self, prompt):
            await asyncio.sleep(5)
            return '{"conflict": true}'

        def _write_pmk_conflict_to_episodic(self, state, reason):
            raise AssertionError("超时不应写 episodic")

    engine = _Engine()
    asyncio.run(engine_reflect.EngineReflect(engine)._pmk_llm_advisory())
    assert engine._speculator_hint == ""


def test_pmk_llm_advisory_skips_without_epistemic_pair(monkeypatch):
    """只有 <2 路认识论文本时不调 LLM（persona 不交给 LLM）."""
    import asyncio

    from huginn.autoloop.engine_reflect import EngineReflect

    monkeypatch.setenv("HUGINN_PMK_INJECT", "1")

    class _Engine:
        _last_pmk_state = {"persona": "recommend GNN", "memory": "use GNN"}
        _speculator_hint = ""

        async def _llm_chat(self, prompt):
            raise AssertionError("不应调用 LLM")

    engine = _Engine()
    asyncio.run(EngineReflect(engine)._pmk_llm_advisory())
    assert engine._speculator_hint == ""


def test_pmk_cross_worldview_is_advisory_not_blocking():
    """世界模型与 kb 的 worldview 标签不兼容 → advisory，绝不阻断（I49）."""
    from huginn.runtime.task_lifecycle import _check_pmk_axes

    axes = _check_pmk_axes({
        "world_model": (
            "world_model=law.exoplanet worldview=physics_causal "
            "domain=exoplanet contract=v1 prediction_error=na"
        ),
        "kb": "T_eq=200 [worldview=latent_transition domain=exoplanet contract=v1]",
    })
    assert not axes.blocked
    assert "worldview_mismatch" in axes.advisory, axes.advisory

    axes2 = _check_pmk_axes({
        "world_model": (
            "world_model=law.exoplanet worldview=physics_causal "
            "domain=exoplanet contract=v1"
        ),
        "memory": "T_eq=210 [worldview=physics_causal domain=mechanics contract=v1]",
    })
    assert not axes2.blocked
    assert "contract_violation" in axes2.advisory, axes2.advisory


def test_pmk_world_model_stalk_advisory():
    """世界模型 stalk：缺标签或高预测误差只 advisory，不阻断."""
    from huginn.runtime.task_lifecycle import _check_pmk_axes, _check_pmk_consistency

    labeled = {
        "world_model": (
            "world_model=law.exoplanet worldview=physics_causal "
            "domain=exoplanet contract=v1 prediction_error=0.30"
        )
    }
    axes = _check_pmk_axes(labeled)
    assert not axes.blocked
    assert "world_model" in axes.advisory
    blocked, reason = _check_pmk_consistency(labeled)
    assert not blocked
    assert "world_model" in reason

    missing = {"world_model": "predicted T_eq_K=200"}
    axes2 = _check_pmk_axes(missing)
    assert not axes2.blocked
    assert "missing labels" in axes2.advisory


def test_build_pmk_state_includes_world_model_stalk():
    from huginn.autoloop.cognitive_guard import build_pmk_state

    class _Eval:
        attempted = "run experiment"
        pmk_feedback = "memory: m"

    state = build_pmk_state(
        None,
        _Eval(),
        None,
        world_model_ctx="world_model=x worldview=physics domain=d contract=v1",
    )
    assert state is not None
    assert state["world_model"].startswith("world_model=")
    assert state["action"] == "run experiment"


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
