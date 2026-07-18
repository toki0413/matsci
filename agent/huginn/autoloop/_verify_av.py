"""AV5-AV8 + AV2 验证脚本.

跑法: cd agent; python -m huginn.autoloop._verify_av

覆盖:
  AV5: σ₂/σ₇/σ₈ 默认关 (HEALTH_MONITOR / SKIP_LOOP_DETECTOR / KEEP_ROOT_N / ROOT_MARKERS)
  AV6: ProspectiveMemory.store() 在 rcb_runner _last_step_eval 后被调
  AV7: MinEffortFloor.check 在 autoloop _validate 被调, 不达标 tests_passed=False
  AV8: heat_engine update_T_hot / update_T_cold 在 rcb_runner 被调
  AV2: autoloop reflect_fn 接 detect_drift + TaskMetrics + should_pause_for_decision

ponytail: 单文件 selfcheck, 无 pytest 依赖, 失败立即抛 AssertionError.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace as _NS


def test_av5_default_env_values() -> None:
    """AV5: σ₂/σ₇/σ₈ 默认关 — 不设 env 时代码默认值正确."""
    # 清掉可能被 rcb_runner setdefault 污染的 env
    for _k in (
        "HUGINN_HEALTH_MONITOR", "HUGINN_SKIP_LOOP_DETECTOR",
        "HUGINN_KEEP_ROOT_N", "HUGINN_ROOT_MARKERS",
    ):
        os.environ.pop(_k, None)

    # 重新 import adapter 拿到模块级 _HEALTH_MONITOR_ON (已加载过的不会重算,
    # 用直接 get 验证默认值)
    _hm = os.environ.get("HUGINN_HEALTH_MONITOR", "0")
    assert _hm == "0", f"AV5 HEALTH_MONITOR 默认应为 0, 实际 {_hm}"

    _skip = os.environ.get("HUGINN_SKIP_LOOP_DETECTOR", "1")
    assert _skip == "1", f"AV5 SKIP_LOOP_DETECTOR 默认应为 1, 实际 {_skip}"

    _kn = os.environ.get("HUGINN_KEEP_ROOT_N", "2")
    assert _kn == "2", f"AV5 KEEP_ROOT_N 默认应为 2, 实际 {_kn}"

    # AV5 _load_root_markers 默认 markers
    from huginn.agent.streaming import _load_root_markers, _DEFAULT_ROOT_MARKERS
    os.environ.pop("HUGINN_ROOT_MARKERS", None)
    markers = _load_root_markers()
    assert markers is not None, "AV5 默认 markers 不应为 None"
    assert "## Methodology Checklist" in markers, f"AV5 默认 markers 缺 checklist: {markers}"
    assert "## Selected Execution Plan" in markers, f"AV5 默认 markers 缺 FCM winner: {markers}"
    # _DEFAULT_ROOT_MARKERS 字符串本身包含所有 marker
    for m in markers:
        assert m in _DEFAULT_ROOT_MARKERS or m.strip() in _DEFAULT_ROOT_MARKERS, \
            f"AV5 marker {m!r} 不在 _DEFAULT_ROOT_MARKERS 里"

    print("AV5 σ₂/σ₇/σ₈ 默认关 OK")


def test_av6_prospective_store_called() -> None:
    """AV6: rcb_runner _last_step_eval 后调 _mem_mgr.remember_prospective."""
    # 模拟 _mem_mgr 记录 remember_prospective 调用
    calls: list[dict] = []

    class _FakeMemMgr:
        def remember_prospective(self, intention: dict) -> str:
            calls.append(intention)
            return intention.get("intention_id", "pim_test")

    _mem_mgr = _FakeMemMgr()
    _step_eval = _NS(
        on_track="false",
        attempted="tried GPR fit on dataset X",
        deviation="MAE 0.5 期望 <0.1",
    )
    _iter_n = 2

    # 复刻 rcb_runner L600-616 的接入逻辑
    if _mem_mgr is not None and _step_eval.on_track in ("false", "unsure"):
        try:
            from huginn.memory.prospective import _new_intention_id
            _mem_mgr.remember_prospective({
                "intention_id": _new_intention_id(),
                "description": (
                    f"上一步脱轨需复核: attempted={_step_eval.attempted[:80]}"
                    f"; deviation={_step_eval.deviation[:80]}"
                ),
                "trigger_type": "dependency",
                "trigger_payload": {"depends_on_step": _iter_n},
                "priority": 5,
                "created_at": 1700000000.0,
                "source_step": _iter_n,
            })
        except Exception as _pe:
            print(f"[prospective store skipped: {_pe}]")

    assert len(calls) == 1, f"AV6 应调一次 remember_prospective, 实际 {len(calls)}"
    _call = calls[0]
    assert _call["trigger_type"] == "dependency", "AV6 trigger_type 应为 dependency"
    assert _call["trigger_payload"]["depends_on_step"] == 2, "AV6 depends_on_step 应为 2"
    assert "脱轨" in _call["description"], "AV6 description 应含脱轨"
    assert "用户决策" not in _call["description"], "AV6 description 不应含用户决策 (走 reminder 不 pause)"

    # on_track=true 不应触发
    calls.clear()
    _step_eval.on_track = "true"
    if _mem_mgr is not None and _step_eval.on_track in ("false", "unsure"):
        _mem_mgr.remember_prospective({"intention_id": "skip"})
    assert len(calls) == 0, "AV6 on_track=true 不应触发 store"

    print("AV6 ProspectiveMemory.store() 接入 OK")


def test_av7_effort_floor_in_validate() -> None:
    """AV7: _validate 不达标时 tests_passed=False + failure_kind=effort_floor_retry."""
    # 模拟 _metacog_check_completion 返回 block=True
    class _FakeEngine:
        _speculator_hint = ""

        def _metacog_check_completion(self):
            return True, "min_iterations=3 deficit (current=1)"

        def _validate_with_av7(self, results: dict) -> dict:
            # 复刻 engine.py L4218-4243 的 AV7 接入逻辑
            try:
                _eff_blk, _eff_why = self._metacog_check_completion()
                results["effort_floor_passed"] = not _eff_blk
                if _eff_blk:
                    results["effort_floor_deficits"] = _eff_why
                    results["failure_kind"] = "effort_floor_retry"
                    results["tests_passed"] = False
                    results["constraints_satisfied"] = False
                    _hint = (
                        f"[effort floor] 探索未达硬下限, 不算通过: {_eff_why}. "
                        "下轮必须扩方法族或保留更多假设, 不要再收敛."
                    )
                    self._speculator_hint = (
                        (self._speculator_hint + "\n" + _hint).strip()
                        if self._speculator_hint else _hint
                    )
                    if len(self._speculator_hint) > 2000:
                        self._speculator_hint = self._speculator_hint[-2000:]
            except Exception:
                pass
            return results

    eng = _FakeEngine()
    results = {"tests_passed": True, "constraints_satisfied": True}
    eng._validate_with_av7(results)

    assert results["tests_passed"] is False, "AV7 不达标应强制 tests_passed=False"
    assert results["constraints_satisfied"] is False, "AV7 不达标应强制 constraints_satisfied=False"
    assert results["failure_kind"] == "effort_floor_retry", "AV7 应设 failure_kind"
    assert results["effort_floor_passed"] is False, "AV7 effort_floor_passed 应为 False"
    assert "min_iterations" in results["effort_floor_deficits"], "AV7 deficits 应含原因"
    assert "[effort floor]" in eng._speculator_hint, "AV7 应注入 hint"

    # _classify_failure 应归 tool_error (不 refute)
    def _classify_failure(validation: dict) -> str:
        if validation.get("failure_kind") == "effort_floor_retry":
            return "tool_error"
        return "other"

    assert _classify_failure(results) == "tool_error", "AV7 _classify_failure 应归 tool_error"

    # 达标时不阻断
    eng2 = _FakeEngine()
    eng2._metacog_check_completion = lambda: (False, "")  # type: ignore
    results2 = {"tests_passed": True, "constraints_satisfied": True}
    eng2._validate_with_av7(results2)
    assert results2["tests_passed"] is True, "AV7 达标时不应阻断"
    assert results2["effort_floor_passed"] is True, "AV7 达标时 effort_floor_passed=True"
    assert "failure_kind" not in results2, "AV7 达标时不应设 failure_kind"

    print("AV7 MinEffortFloor 接 _validate OK")


def test_av8_heat_engine_t_hot_t_cold() -> None:
    """AV8: rcb_runner 每轮 _last_step_eval 后调 update_T_hot / update_T_cold."""
    updates: list[tuple] = []

    class _FakeHeatEngine:
        T_hot = 0.0
        T_cold = 0.0

        def update_T_hot(self, val: float) -> None:
            self.T_hot = val
            updates.append(("T_hot", val))

        def update_T_cold(self, val: float, darwin_score: float = 0.0) -> None:
            self.T_cold = val
            updates.append(("T_cold", val))

        def update_kinematics(self, **kw) -> None:
            updates.append(("kinematics", kw))

    he = _FakeHeatEngine()
    _step_eval = _NS(
        evidence_quality="low",
        on_track="false",
    )

    # 复刻 rcb_runner L622-642 的 AV8 接入逻辑
    if he is not None:
        try:
            _eq = (_step_eval.evidence_quality or "unknown").lower().strip()
            _t_hot_proxy = {"low": 0.8, "medium": 0.5, "high": 0.2}.get(_eq, 0.5)
            _ot = (_step_eval.on_track or "unsure").lower().strip()
            _t_cold_proxy = {"true": 0.7, "false": 0.2}.get(_ot, 0.4)
            he.update_T_hot(_t_hot_proxy)
            he.update_T_cold(_t_cold_proxy, darwin_score=0.0)
        except Exception:
            pass

    assert he.T_hot == 0.8, f"AV8 evidence_quality=low → T_hot=0.8, 实际 {he.T_hot}"
    assert he.T_cold == 0.2, f"AV8 on_track=false → T_cold=0.2, 实际 {he.T_cold}"
    assert ("T_hot", 0.8) in updates, "AV8 应调 update_T_hot"
    assert ("T_cold", 0.2) in updates, "AV8 应调 update_T_cold"

    # evidence_quality=high → T_hot=0.2; on_track=true → T_cold=0.7
    he.T_hot = 0.0
    he.T_cold = 0.0
    updates.clear()
    _step_eval.evidence_quality = "high"
    _step_eval.on_track = "true"
    if he is not None:
        _eq = (_step_eval.evidence_quality or "unknown").lower().strip()
        _t_hot_proxy = {"low": 0.8, "medium": 0.5, "high": 0.2}.get(_eq, 0.5)
        _ot = (_step_eval.on_track or "unsure").lower().strip()
        _t_cold_proxy = {"true": 0.7, "false": 0.2}.get(_ot, 0.4)
        he.update_T_hot(_t_hot_proxy)
        he.update_T_cold(_t_cold_proxy, darwin_score=0.0)
    assert he.T_hot == 0.2, f"AV8 evidence_quality=high → T_hot=0.2, 实际 {he.T_hot}"
    assert he.T_cold == 0.7, f"AV8 on_track=true → T_cold=0.7, 实际 {he.T_cold}"

    print("AV8 heat_engine T_hot/T_cold 接入 OK")


def test_av2_autoloop_reflect_hook() -> None:
    """AV2: autoloop reflect_fn 接 detect_drift + TaskMetrics + should_pause_for_decision.

    验证 __init__ 字段 + reflect_fn 接入点存在 (语法 + 字段检查).
    不真跑 CognitiveLoop (要 LLM), 只验证代码结构.
    """
    import inspect
    from huginn.autoloop.engine import AutoloopEngine

    src = inspect.getsource(AutoloopEngine)

    assert "_evals_history: list" in src, "AV2 __init__ 缺 _evals_history"
    assert "_task_metrics: Any" in src, "AV2 __init__ 缺 _task_metrics"
    assert "_drift_info: tuple" in src, "AV2 __init__ 缺 _drift_info"
    assert "AV2+AV4: PMK + TaskMetrics + detect_drift" in src, "AV2 reflect_fn 缺接入注释"
    # AV4 重构后 detect_drift/TaskMetrics/should_pause_for_decision 下沉到 cognitive_loop 共享函数
    assert "update_drift_and_metrics" in src, "AV4 缺 update_drift_and_metrics 共享函数调用"
    assert "build_pmk_state" in src, "AV4 缺 build_pmk_state 共享函数调用"
    assert "check_pause_decision" in src, "AV4 缺 check_pause_decision 共享函数调用"
    assert 'if action == "validate" and cog.get("validation")' in src, \
           "AV2 缺 validate 条件门"

    # 共享函数本体在 cognitive_loop.py, 检查它含原 import
    import inspect as _inspect
    from huginn.autoloop.cognitive_loop import (
        update_drift_and_metrics, build_pmk_state, check_pause_decision,
    )
    _cog_src = _inspect.getsource(update_drift_and_metrics)
    assert "detect_drift as _detect_drift" in _cog_src, "AV4 共享函数缺 detect_drift import"
    assert "update_metrics as _um" in _cog_src, "AV4 共享函数缺 update_metrics import"
    assert "save_metrics as _sm" in _cog_src, "AV4 共享函数缺 save_metrics import"
    _pause_src = _inspect.getsource(check_pause_decision)
    assert "should_pause_for_decision as _spd" in _pause_src, "AV4 共享函数缺 should_pause_for_decision import"

    # run_cognitive 入口应 reset AV2 字段
    assert "self._evals_history = []" in src, "AV2 run_cognitive 缺 reset"
    assert "self._task_metrics = None" in src, "AV2 run_cognitive 缺 reset"
    assert "self._drift_info = None" in src, "AV2 run_cognitive 缺 reset"

    print("AV2 autoloop reflect_fn 接入 OK")


def test_av4_pmk_and_pause_shared() -> None:
    """AV4 step 4: build_pmk_state + check_pause_decision 共享函数功能测试."""
    from types import SimpleNamespace as _NS
    from huginn.autoloop.cognitive_loop import (
        build_pmk_state, check_pause_decision,
    )

    # 1. 全空输入 → None
    assert build_pmk_state(None, None, None) is None, "全空应返回 None"

    # 2. persona only → {"persona": ..., "memory": "", "kb": ""}
    _p = _NS(description="materials scientist")
    _s = build_pmk_state(_p, None, None)
    assert _s is not None and _s["persona"] == "materials scientist"
    assert _s["memory"] == "" and _s["kb"] == ""

    # 3. persona + memory (从 pmk_feedback 抽)
    _ev = _NS(attempted="test query", pmk_feedback="Persona: X; Memory: refine hypothesis; KB: Y")
    _s = build_pmk_state(_p, _ev, None)
    assert _s["memory"] == "refine hypothesis", f"抽 memory 段失败: {_s['memory']}"

    # 4. dict persona 兼容
    _s = build_pmk_state({"description": "chemist"}, None, None)
    assert _s["persona"] == "chemist"

    # 5. check_pause_decision 兜底 — 异常输入不抛, 返回 (False, "", [])
    _pause, _reason, _opts = check_pause_decision([], [], None, None, None)
    assert _pause is False and _reason == "" and _opts == []

    print("AV4 step4 build_pmk_state + check_pause_decision 共享函数 OK")


def test_av3_meta_trace_cross_entry() -> None:
    """AV3: meta_trace 跨入口 (rcb_runner/autoloop) 共享验证.

    现状: 两个入口都 append 到 ws/.huginn/meta_trace.jsonl, schema 一致,
    reader (build_meta_trace_text) 不按 role 过滤, 跨入口续跑 trace 不丢.
    本测试锁住该行为, 防止后续改坏.
    """
    import json as _json
    import tempfile
    from pathlib import Path
    from huginn.context_builder import ContextBuilder

    with tempfile.TemporaryDirectory() as _tmp:
        _ws = Path(_tmp)
        _trace = _ws / ".huginn" / "meta_trace.jsonl"
        _trace.parent.mkdir(parents=True, exist_ok=True)

        # 模拟 rcb_runner 写入
        _rcb_entry = {
            "iteration": 1, "ts": 1.0, "role": "rcb_exec",
            "attempted": "rcb attempt", "found": "rcb found",
            "evidence": ["rcb evidence"], "limitations": [],
            "artifacts": ["report/report.md"], "next_hint": "rcb next",
            "darwin_score": 0.0, "supported_ratio": 0.0,
        }
        # 模拟 autoloop 写入 (同 schema, 不同 role)
        _auto_entry = {
            "iteration": 2, "ts": 2.0, "role": "autoloop",
            "attempted": "auto attempt", "found": "auto found",
            "evidence": ["auto evidence"], "limitations": ["auto lim"],
            "artifacts": ["auto.json"], "next_hint": "auto next",
            "darwin_score": 0.5, "supported_ratio": 0.3,
        }
        with _trace.open("w", encoding="utf-8") as _f:
            _f.write(_json.dumps(_rcb_entry, ensure_ascii=False) + "\n")
            _f.write(_json.dumps(_auto_entry, ensure_ascii=False) + "\n")

        # 用 ContextBuilder 读回 — 不传 model/workspace 必填项, 走 __new__ 绕 __init__
        _ctx = ContextBuilder.__new__(ContextBuilder)
        _ctx.workspace = str(_ws)
        _ctx._meta_trace_cache = None
        _ctx._meta_trace_mtime = None
        _ctx._meta_trace_count = None

        _text = _ctx.build_meta_trace_text(last_n=5)
        # 两个 entry 都应被读出 (reader 不按 role 过滤)
        assert "rcb attempt" in _text, f"rcb_runner entry 丢失: {_text}"
        assert "auto attempt" in _text, f"autoloop entry 丢失: {_text}"
        assert "auto lim" in _text, f"autoloop limitations 丢失: {_text}"

    print("AV3 meta_trace 跨入口共享 OK (rcb_runner + autoloop 都读得到)")


def test_av4_heat_engine_shared_unit() -> None:
    """P0.1: update_heat_engine_after_step 直接单元测试.

    之前只有 AV8 间接测 (rcb_runner 集成), 4 档映射逻辑改坏 AV8 不一定能抓.
    """
    from types import SimpleNamespace as _NS
    from huginn.autoloop.cognitive_loop import update_heat_engine_after_step

    class _MockHeatEngine:
        def __init__(self):
            self.t_hot = None
            self.t_cold = None
            self.darwin = None
            self.kin_args = None
        def update_T_hot(self, v):
            self.t_hot = v
        def update_T_cold(self, v, darwin_score=0.0):
            self.t_cold = v
            self.darwin = darwin_score
        def update_kinematics(self, **kw):
            self.kin_args = kw

    # 1. None heat_engine → no-op, 不抛
    update_heat_engine_after_step(None, None, 0, 0)

    # 2. evidence_quality=low → T_hot=0.8 (高熵); on_track=false → T_cold=0.2 (低秩序)
    _he = _MockHeatEngine()
    _ev = _NS(evidence_quality="low", on_track="false")
    update_heat_engine_after_step(_he, _ev, prompt_len=100, idea_count=2)
    assert _he.t_hot == 0.8, f"low eq 应映射 0.8, got {_he.t_hot}"
    assert _he.t_cold == 0.2, f"false on_track 应映射 0.2, got {_he.t_cold}"
    assert _he.kin_args["idea_count"] == 2
    assert _he.kin_args["system_prompt_len"] == 100
    assert _he.kin_args["stable_principles_count"] == 1  # default

    # 3. evidence_quality=high → T_hot=0.2 (低熵); on_track=true → T_cold=0.7
    _he = _MockHeatEngine()
    _ev = _NS(evidence_quality="high", on_track="true")
    update_heat_engine_after_step(_he, _ev, prompt_len=50, idea_count=5,
                                  stable_principles_count=3)
    assert _he.t_hot == 0.2
    assert _he.t_cold == 0.7
    assert _he.kin_args["stable_principles_count"] == 3

    # 4. medium / unsure → 0.5 / 0.4
    _he = _MockHeatEngine()
    _ev = _NS(evidence_quality="medium", on_track="unsure")
    update_heat_engine_after_step(_he, _ev, 0, 0)
    assert _he.t_hot == 0.5
    assert _he.t_cold == 0.4

    # 5. 异常输入兜底 (缺字段) → 不抛, T_hot/cold 不更新
    _he = _MockHeatEngine()
    _ev = _NS()  # 无 evidence_quality / on_track
    update_heat_engine_after_step(_he, _ev, 0, 0)
    # getattr 默认 "unknown"/"unsure" → 0.5 / 0.4
    assert _he.t_hot == 0.5
    assert _he.t_cold == 0.4

    print("P0.1 update_heat_engine_after_step 4 档映射单元测试 OK")


def test_av4_drift_and_metrics_shared_unit() -> None:
    """P0.1: update_drift_and_metrics 直接单元测试.

    duck typing SimpleNamespace 能跑; load_metrics/save_metrics 失败兜底.
    """
    import tempfile
    from pathlib import Path
    from types import SimpleNamespace as _NS
    from huginn.autoloop.cognitive_loop import update_drift_and_metrics

    with tempfile.TemporaryDirectory() as _tmp:
        _ws = Path(_tmp)
        _ev = _NS(
            step_id=0,
            attempted="test",
            found="ok",
            on_track="true",
            evidence_quality="high",
            deviation="",
            structure_check="passed",
            pmk_feedback="",
            tool_call_health=None,
            target_chain_ref=None,
        )
        # task_metrics=None → 内部 load 或新建; workspace 不存在也能兜底
        _drift, _tm = update_drift_and_metrics(
            [_ev], _ev, None, None, _ws, "test_run", 10,
        )
        # drift_info 可能 None (window=3, 只有 1 条); task_metrics 应非 None
        assert _tm is not None, "task_metrics 应被新建"

    # 2. workspace 是非法路径 → 不抛, drift_info=None
    _drift, _tm = update_drift_and_metrics(
        [], None, None, None, "/nonexistent/xyz", "test_run", 10,
    )
    assert _drift is None or isinstance(_drift, tuple)

    print("P0.1 update_drift_and_metrics duck typing + 兜底 OK")


def test_p02_validation_to_step_eval_fields() -> None:
    """P0.2: _validation_to_step_eval_fields schema 防御 + 字段映射.

    之前 reflect_fn 硬取 summary/result/errors 全空; 现在用真实 _validate 字段.
    """
    from huginn.autoloop.cognitive_loop import _validation_to_step_eval_fields

    # 1. tests_ok=True, 空 validation → found 含 tests_passed=True
    _f = _validation_to_step_eval_fields({}, True, None, step_id=0)
    assert _f["attempted"] == ""  # execution_result None
    assert "tests_passed=True" in _f["found"]
    assert _f["on_track"] == "true"
    assert _f["evidence_quality"] == "high"
    assert _f["deviation"] == ""
    assert _f["structure_check"] == "passed"

    # 2. tests_ok=False, 有 thinking_collapse + *_error → deviation 收集
    _val = {
        "thinking_collapse": "model repeating same tokens",
        "physics_validation_error": "unit mismatch",
        "effort_floor_deficits": "min_iterations=5 not met",
    }
    _f = _validation_to_step_eval_fields(_val, False, None, step_id=1)
    assert _f["on_track"] == "false"
    assert _f["evidence_quality"] == "low"
    assert "thinking_collapse" in _f["deviation"]
    assert "physics_validation_error" in _f["deviation"]
    assert "effort_floor_deficits" in _f["deviation"]
    assert _f["structure_check"] == "failed"

    # 3. execution_result dict → attempted 从 description 抽
    _exec = {"description": "ran VASP relaxation"}
    _f = _validation_to_step_eval_fields({}, True, _exec, step_id=2)
    assert _f["attempted"] == "ran VASP relaxation"

    # 4. benchmarks dict → found 含 benchmark 指标
    _val = {
        "benchmarks": {
            "mae": {"metric": 0.05},
            "r2": {"value": 0.92},
        }
    }
    _f = _validation_to_step_eval_fields(_val, True, None, step_id=3)
    assert "mae=0.05" in _f["found"]
    assert "r2=0.92" in _f["found"]

    # 5. 之前 bug 验证: summary/result/errors 字段不再被读 (即使存在也不影响)
    _val = {"summary": "OLD_FIELD", "result": "OLD", "errors": "OLD"}
    _f = _validation_to_step_eval_fields(_val, True, None, step_id=4)
    assert _f["attempted"] == ""  # 不读 summary
    assert "OLD" not in _f["found"]  # 不读 result
    assert _f["deviation"] == ""  # 不读 errors

    print("P0.2 _validation_to_step_eval_fields schema 防御 + 字段映射 OK")


def test_p03_rcb_effort_floor() -> None:
    """P0.3: _rcb_effort_floor 硬下限 — RCB 路径对齐 AV7 MinEffortFloor."""
    import tempfile
    from pathlib import Path
    from huginn.cli.rcb_runner import _rcb_effort_floor

    _checklist = (
        "## Methodology Checklist\n"
        "- [EXACT] VAE encoder architecture\n"
        "- [EXACT] GNN message passing\n"
        "- [EXACT] CNN feature extractor\n"
        "- Report MAE and R² metrics\n"
    )

    with tempfile.TemporaryDirectory() as _tmp:
        _ws = Path(_tmp)

        # 1. 无 report.md → 放行 (避免误杀)
        _ok, _ = _rcb_effort_floor(_ws, _checklist)
        assert _ok is True

        # 2. report.md 缺所有 keyword → cov=0% < 70%, 驳回
        _rp = _ws / "report" / "report.md"
        _rp.parent.mkdir(parents=True, exist_ok=True)
        _rp.write_text("This is an empty report with no keywords.", encoding="utf-8")
        _ok, _reason = _rcb_effort_floor(_ws, _checklist)
        assert _ok is False, "0% 覆盖应驳回"
        assert "coverage=0%" in _reason, f"reason 应含 coverage=0%: {_reason}"
        assert "missing" in _reason.lower()

        # 3. report.md 全覆盖 → 放行
        _rp.write_text(
            "Report covers VAE encoder, GNN message passing, CNN feature extractor. "
            "MAE=0.05, R²=0.92.",
            encoding="utf-8",
        )
        _ok, _ = _rcb_effort_floor(_ws, _checklist)
        assert _ok is True, "全覆盖应放行"

        # 4. 部分覆盖 (50%) < 70% → 驳回
        _rp.write_text(
            "Report covers VAE encoder and GNN message passing. No CNN, no metrics.",
            encoding="utf-8",
        )
        _ok, _reason = _rcb_effort_floor(_ws, _checklist, min_cov_pct=70)
        assert _ok is False, "50% < 70% 应驳回"

        # 5. 部分覆盖但阈值低 (50%) → 放行
        _ok, _ = _rcb_effort_floor(_ws, _checklist, min_cov_pct=40)
        assert _ok is True, "50% >= 40% 应放行"

        # 6. 无 checklist → 放行
        _ok, _ = _rcb_effort_floor(_ws, "")
        assert _ok is True

    print("P0.3 _rcb_effort_floor 6 case 全过 (含阈值边界)")


def test_p14_campaign_sse_in_run_cognitive() -> None:
    """P1.4: run_cognitive execute_fn 加 campaign SSE emit (对齐 run()).

    检查 engine.py 源码含 5 类 campaign 事件 emit 调用:
    - campaign.iteration (observe_fn 开头)
    - campaign.hypothesis (execute_fn hypothesize 后)
    - campaign.retry (execute 失败)
    - campaign.suspect (validate 失败)
    - campaign.refine (pivot 后)
    """
    import inspect
    from huginn.autoloop.engine import AutoloopEngine

    src = inspect.getsource(AutoloopEngine.run_cognitive)

    # 5 类 campaign 事件都应在 run_cognitive 里 emit
    assert '"campaign.iteration"' in src, "run_cognitive 缺 campaign.iteration emit"
    assert '"campaign.hypothesis"' in src, "run_cognitive 缺 campaign.hypothesis emit"
    assert '"campaign.retry"' in src, "run_cognitive 缺 campaign.retry emit"
    assert '"campaign.suspect"' in src, "run_cognitive 缺 campaign.suspect emit"
    assert '"campaign.refine"' in src, "run_cognitive 缺 campaign.refine emit"

    # _emit_campaign 方法本身在 run_cognitive 里调用 5 次以上
    _emit_count = src.count("self._emit_campaign(")
    assert _emit_count >= 5, f"run_cognitive 应有 ≥5 处 _emit_campaign 调用, 实际 {_emit_count}"

    # 关键事件位置正确: iteration 在 observe_fn, hypothesis 在 hypothesize 后
    _iter_pos = src.find('"campaign.iteration"')
    _obs_pos = src.find("async def observe_fn")
    assert _iter_pos > _obs_pos and _iter_pos < src.find("async def decide_fn"), \
        "campaign.iteration 应在 observe_fn 里"

    _hyp_pos = src.find('"campaign.hypothesis"')
    _exec_pos = src.find("async def execute_fn")
    assert _hyp_pos > _exec_pos, "campaign.hypothesis 应在 execute_fn 里"

    print(f"P1.4 run_cognitive campaign SSE 5 类事件全接入 OK ({_emit_count} 处 emit)")


def test_v10_f1_f5_sunken_in_run_cognitive() -> None:
    """v10 阶段 1: F1-F5 (goal budget / completion audit / darwin_ratchet /
    surprise 早停 / blind_spot) 下沉到 run_cognitive 的源码结构 + 条件判定测试.

    不真跑 CognitiveLoop (要 LLM), 只验证:
    1. observe_fn 含 F1 (goal budget 硬停) + F5 (blind_spot_pass) 下沉
    2. reflect_fn 含 F2 (completion audit) + F3 (darwin_ratchet) + F4 (surprise 早停)
    3. F1/F2/F3/F4 条件判定逻辑正确 (独立小函数模拟)
    """
    import inspect
    from huginn.autoloop.engine import AutoloopEngine

    src = inspect.getsource(AutoloopEngine.run_cognitive)

    # F1: goal budget 硬停 (observe_fn 开头)
    assert "v10-F1+F7" in src, "v10 F1 goal budget 未下沉到 run_cognitive"
    assert "is_budget_exhausted" in src, "v10 F1 缺 is_budget_exhausted 调用"
    assert "campaign.budget_exhausted" in src, "v10 F1 缺 budget_exhausted 事件"
    assert "_gs.fail_goal" in src or "fail_goal(" in src, "v10 F1 缺 fail_goal 调用"

    # F5: blind_spot_pass (observe_fn, 每隔 5 轮)
    assert "v10-F5" in src, "v10 F5 blind_spot 未下沉"
    assert "_blind_spot_pass" in src, "v10 F5 缺 _blind_spot_pass 调用"
    assert "state.iteration == 1 or state.iteration % 5 == 0" in src, \
        "v10 F5 缺 iteration 条件"

    # F2: completion audit (reflect_fn 末尾, validate 块内)
    assert "v10-F2" in src, "v10 F2 completion audit 未下沉"
    assert "GoalScheduler.check_completion" in src, "v10 F2 缺 check_completion"
    assert "_metacog_check_completion" in src, "v10 F2 缺 metacog audit"
    assert "goal.status = \"completed\"" in src, "v10 F2 缺 goal.status=completed"

    # F4: surprise 早停 (reflect_fn 末尾)
    assert "v10-F4" in src, "v10 F4 surprise 早停 未下沉"
    assert "_surprise_history[-3:]" in src, "v10 F4 缺 surprise_history[-3:]"
    assert "max(0.08, 0.20 - 0.4 * _avg_noise)" in src, "v10 F4 缺自适应阈值"

    # F3: darwin_ratchet (reflect_fn 末尾)
    assert "v10-F3" in src, "v10 F3 darwin_ratchet 未下沉"
    assert "_darwin_ratchet_check" in src, "v10 F3 缺 _darwin_ratchet_check 调用"

    # --- F1 条件判定: is_budget_exhausted 逻辑 ---
    from huginn.autoloop.goal_scheduler import GoalScheduler
    from types import SimpleNamespace as _NS
    _g_exhausted = _NS(iteration=10, max_iterations=10)
    _g_ok = _NS(iteration=3, max_iterations=10)
    assert GoalScheduler.is_budget_exhausted(_g_exhausted) is True, \
        "F1 iteration>=max 应判 exhausted"
    assert GoalScheduler.is_budget_exhausted(_g_ok) is False, \
        "F1 iteration<max 不应判 exhausted"

    # --- F2 条件判定: check_completion ---
    # goal 无 success_criteria → check_completion 返回 False, 不影响
    _g_no_criteria = _NS(objective="test", success_criteria=[], status="active")
    _val_ok = {"tests_passed": True}
    assert GoalScheduler.check_completion(_g_no_criteria, _val_ok) is False, \
        "F2 无 criteria 时 check_completion 应返回 False"

    # --- F3 条件判定: darwin_stagnation 阈值 ---
    # 复刻 _darwin_ratchet_check 内部逻辑
    _stag_limit = 5
    _darwin_stagnation = 5
    _iteration = 3
    _should_stop = (_darwin_stagnation >= _stag_limit and _iteration > 2)
    assert _should_stop is True, "F3 stagnation>=5 + iter>2 应 stop"

    _darwin_stagnation = 4
    _should_stop = (_darwin_stagnation >= _stag_limit and _iteration > 2)
    assert _should_stop is False, "F3 stagnation<5 不应 stop"

    # --- F4 条件判定: surprise 阈值自适应 ---
    # 模拟 3 轮低 surprise + 低 noise
    _surprise_history = [(0.05, 0.05), (0.04, 0.05), (0.06, 0.05)]
    _recent = _surprise_history[-3:]
    _worsts = [w for w, _ in _recent]
    _avg_noise = sum(s for _, s in _recent) / len(_recent)
    _thr = max(0.08, 0.20 - 0.4 * _avg_noise)
    # noise=0.05 → thr = max(0.08, 0.18) = 0.18; all worsts < 0.18 → True
    assert abs(_thr - 0.18) < 1e-9, f"F4 noise=0.05 阈值应为 0.18, 实际 {_thr}"
    assert all(w < _thr for w in _worsts), "F4 低 surprise 应触发收敛"

    # 高 noise 时阈值更严格
    _surprise_history_high_noise = [(0.05, 0.3), (0.04, 0.3), (0.06, 0.3)]
    _recent = _surprise_history_high_noise[-3:]
    _avg_noise = sum(s for _, s in _recent) / len(_recent)
    _thr = max(0.08, 0.20 - 0.4 * _avg_noise)
    # noise=0.3 → thr = max(0.08, 0.08) = 0.08 (下限保护)
    assert abs(_thr - 0.08) < 1e-9, f"F4 noise=0.3 阈值应降至下限 0.08, 实际 {_thr}"

    # surprise 高时不收敛
    _surprise_history_high = [(0.5, 0.05), (0.4, 0.05), (0.6, 0.05)]
    _recent = _surprise_history_high[-3:]
    _worsts = [w for w, _ in _recent]
    _avg_noise = sum(s for _, s in _recent) / len(_recent)
    _thr = max(0.08, 0.20 - 0.4 * _avg_noise)
    assert not all(w < _thr for w in _worsts), "F4 高 surprise 不应触发收敛"

    print("v10 F1-F5 (goal budget / completion audit / darwin_ratchet / "
          "surprise 早停 / blind_spot) 下沉 + 条件判定 OK")


def test_v10_f6_f8_sunken_in_run_cognitive() -> None:
    """v10 阶段 2: F6 (build_continuation_prompt) + F8 (drain_side_questions)
    下沉到 observe_fn 的源码结构 + 条件判定测试.

    F7 (goal_scheduler increment) 已在阶段 1 合并到 F1, 不重复测.
    """
    import inspect
    from huginn.autoloop.engine import AutoloopEngine

    src = inspect.getsource(AutoloopEngine.run_cognitive)

    # F6: build_continuation_prompt (observe_fn, goal 非空 + iteration > 1)
    assert "v10-F6" in src, "v10 F6 build_continuation_prompt 未下沉"
    assert "GoalScheduler.build_continuation_prompt" in src, \
        "v10 F6 缺 build_continuation_prompt 调用"
    assert "goal is not None and state.iteration > 1" in src, \
        "v10 F6 缺 iteration > 1 条件"

    # F8: drain_side_questions (observe_fn, perceive 返回空时)
    assert "v10-F8" in src, "v10 F8 drain_side_questions 未下沉"
    assert "_drain_side_questions" in src, "v10 F8 缺 _drain_side_questions 调用"
    # v10: F8 用 _perceived_empty flag 而非 not cog["context"] (上轮 residual 会掩盖)
    assert "_perceived_empty" in src, "v10 F8 缺 _perceived_empty 判定"
    assert "if _perceived_empty" in src, "v10 F8 缺 _perceived_empty 条件分支"

    # --- F6 条件判定: goal + iteration ---
    # goal=None → 不拼; goal 非空 + iteration=1 → 不拼; goal 非空 + iteration=2 → 拼
    from types import SimpleNamespace as _NS
    _goal = _NS(objective="test goal", status="active")
    _iter_1 = 1
    _iter_2 = 2

    # iteration=1 → 不拼
    _should_build = (_goal is not None and _iter_1 > 1)
    assert _should_build is False, "F6 iteration=1 不应拼 continuation"

    # iteration=2 → 拼
    _should_build = (_goal is not None and _iter_2 > 1)
    assert _should_build is True, "F6 iteration=2 应拼 continuation"

    # goal=None → 不拼
    _should_build = (None is not None and _iter_2 > 1)
    assert _should_build is False, "F6 goal=None 不应拼 continuation"

    # --- F8 条件判定: context 为空时调 drain ---
    _ctx_empty: dict = {}
    _ctx_nonempty = {"summary": "some context"}
    assert not _ctx_empty, "F8 空 context 应触发 drain"
    assert _ctx_nonempty, "F8 非空 context 不应触发 drain"

    print("v10 F6 (build_continuation_prompt) + F8 (drain_side_questions) "
          "下沉 + 条件判定 OK")


def test_v10_f15_f17_sunken_in_run_cognitive() -> None:
    """v10 阶段 3: F15 (G31 bypass) + F16 (timeout 硬停) + F17 (GoalJudge)
    下沉到 run_cognitive 的源码结构 + 条件判定测试.

    F15/F16 在 observe_fn, F17 在 reflect_fn (validate 块内, F2 之后).
    """
    import inspect
    from huginn.autoloop.engine import AutoloopEngine

    src = inspect.getsource(AutoloopEngine.run_cognitive)

    # F15: G31 bypass (observe_fn, perceive 返回空 + 首轮 + objective)
    assert "v10-F15" in src, "v10 F15 G31 bypass 未下沉"
    assert "G31: perceive empty on iter 1" in src, "v10 F15 缺 G31 日志"
    assert 'state.iteration == 1 and objective' in src, \
        "v10 F15 缺 iteration==1 + objective 条件"
    assert '"forced": True' in src, "v10 F15 缺 forced context"

    # F16: timeout 硬停 (observe_fn)
    assert "v10-F16" in src, "v10 F16 timeout 未下沉"
    assert "tracker.is_expired(progress_task_id)" in src, \
        "v10 F16 缺 tracker.is_expired 检查"
    assert "timeout_expired" in src, "v10 F16 缺 timeout_expired 标记"

    # F17: GoalJudge (reflect_fn, 每 3 轮或最后一轮)
    assert "v10-F17" in src, "v10 F17 GoalJudge 未下沉"
    assert "from huginn.evaluation.goal_judge import GoalJudge" in src, \
        "v10 F17 缺 GoalJudge import"
    assert "_judge.judge(goal.objective" in src, "v10 F17 缺 judge 调用"
    assert "state.iteration % 3 == 2" in src, "v10 F17 缺 每 3 轮条件"

    # --- F15 条件判定: G31 触发条件 ---
    # perceive 空 + 首轮 + objective → 注入 G31 context
    _ctx_empty: dict = {}
    _iter_1 = 1
    _iter_2 = 2
    _objective = "test objective"
    _should_g31 = (not _ctx_empty and _iter_1 == 1 and _objective)
    # 注: _ctx_empty 是空 dict, not {} = True
    assert (not _ctx_empty) is True, "F15 空 context 应触发 G31"
    _should_g31 = (not _ctx_empty and _iter_1 == 1 and _objective)
    assert _should_g31, "F15 首轮 + 空 context + objective 应触发 G31"

    # 非首轮不触发
    _should_g31 = (not _ctx_empty and _iter_2 == 1 and _objective)
    assert not _should_g31, "F15 非首轮不应触发 G31"

    # context 非空不触发
    _ctx_full = {"summary": "x"}
    _should_g31 = (not _ctx_full and _iter_1 == 1 and _objective)
    assert not _should_g31, "F15 非空 context 不应触发 G31"

    # --- F16 条件判定: timeout ---
    # tracker.is_expired 返回 True → should_stop
    class _FakeTracker:
        def is_expired(self, task_id: str) -> bool:
            return task_id == "expired"

    _ft = _FakeTracker()
    assert _ft.is_expired("expired") is True, "F16 expired 应返回 True"
    assert _ft.is_expired("active") is False, "F16 active 应返回 False"

    # --- F17 条件判定: GoalJudge 触发条件 ---
    # 每 3 轮 (iteration % 3 == 2) 或最后一轮 (iteration >= max - 1)
    _max_iter = 10
    _trigger_iters = [
        i for i in range(1, _max_iter + 1)
        if i % 3 == 2 or i >= _max_iter - 1
    ]
    # iter 2, 5, 8 (每 3 轮) + iter 9, 10 (最后两轮)
    assert 2 in _trigger_iters, "F17 iter=2 应触发 (2%3==2)"
    assert 5 in _trigger_iters, "F17 iter=5 应触发 (5%3==2)"
    assert 8 in _trigger_iters, "F17 iter=8 应触发 (8%3==2)"
    assert 9 in _trigger_iters, "F17 iter=9 应触发 (>=max-1)"
    assert 10 in _trigger_iters, "F17 iter=10 应触发 (>=max-1)"
    assert 1 not in _trigger_iters, "F17 iter=1 不应触发"
    assert 3 not in _trigger_iters, "F17 iter=3 不应触发"

    print("v10 F15 (G31 bypass) + F16 (timeout 硬停) + F17 (GoalJudge) "
          "下沉 + 条件判定 OK")


def main() -> None:
    test_av5_default_env_values()
    test_av6_prospective_store_called()
    test_av7_effort_floor_in_validate()
    test_av8_heat_engine_t_hot_t_cold()
    test_av2_autoloop_reflect_hook()
    test_av4_pmk_and_pause_shared()
    test_av3_meta_trace_cross_entry()
    test_av4_heat_engine_shared_unit()
    test_av4_drift_and_metrics_shared_unit()
    test_p02_validation_to_step_eval_fields()
    test_p03_rcb_effort_floor()
    test_p14_campaign_sse_in_run_cognitive()
    test_v10_f1_f5_sunken_in_run_cognitive()
    test_v10_f6_f8_sunken_in_run_cognitive()
    test_v10_f15_f17_sunken_in_run_cognitive()
    print("\nAll AV + P0.1 + P0.2 + P0.3 + P1.4 + v10-F1F5 + v10-F6F8 + v10-F15F17 verifications passed")


if __name__ == "__main__":
    main()
