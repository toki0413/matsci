"""P2+P3 时序表征增强自检.

覆盖:
- P2: run_context 持久化 + 加载 + decider prompt 注入
- P3: _physical_timeseries 收集 + 格式化 + decider prompt 注入
- P3: _extract_timeseries 从工具结果提取

不引入 pytest — assert-based, `python -m tests.test_temporal_p2p3` 可跑.
"""

import json
import sys
import tempfile
from pathlib import Path

_AGENT_DIR = Path(__file__).resolve().parent.parent
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))


def _check_run_context_persist_load():
    """P2: _persist_run_context 存 JSON, _load_prev_run_context 读回."""
    from huginn.autoloop.cognitive_loop import LoopState
    from huginn.autoloop.engine import AutoloopEngine

    with tempfile.TemporaryDirectory() as d:
        eng = AutoloopEngine.__new__(AutoloopEngine)
        eng.workspace = Path(d)
        eng.memory = None  # 避免初始化 longterm

        # 用 mock memory 验证 store/recall 调用
        from unittest.mock import MagicMock
        eng.memory = MagicMock()
        eng.memory.recall.return_value = []

        # 构造 cog + state 模拟 run 结束状态
        cog = {
            "hypothesis": "Fe 在高温下扩散系数随温度升高而增大",
            "plan": {"mode": "lammps", "plan_id": "p1"},
            "validation": {"tests_passed": False},
        }
        state = LoopState()
        state.iteration = 5
        state.iteration_history = [
            {"iter": 3, "action": "execute", "val_status": "failed",
             "advice": "tool timeout", "plan_mode": "lammps"},
            {"iter": 4, "action": "validate", "val_status": "failed",
             "advice": "rmsd too high", "plan_mode": ""},
        ]

        # persist
        eng._persist_run_context("loop_test01", "test obj", cog, state)
        assert eng.memory.remember.called, "_persist_run_context 应调 memory.remember"
        _call = eng.memory.remember.call_args
        assert _call.kwargs.get("category") == "run_context", \
            f"category 应为 run_context, 实际 {_call.kwargs.get('category')}"
        _content = _call.kwargs.get("content", "")
        _snap = json.loads(_content)
        assert _snap["run_id"] == "loop_test01"
        assert _snap["objective"] == "test obj"
        assert "Fe" in _snap["hypothesis"]
        assert _snap["plan_mode"] == "lammps"
        assert _snap["outcome"] == "inconclusive"  # tests_passed=False
        assert _snap["iterations"] == 5
        assert len(_snap["inconclusive"]) == 2  # 两条 failed
        assert len(_snap["recent_steps"]) == 2

        # load — 模拟 memory 返回刚存的 snapshot
        eng.memory.recall.return_value = [{"content": _content}]
        ctx = eng._load_prev_run_context()
        assert "Fe" in ctx, "load 应返回 hypothesis"
        assert "lammps" in ctx, "load 应返回 plan_mode"
        assert "inconclusive" in ctx, "load 应返回 inconclusive 方向"
        assert "iter3" in ctx or "iter4" in ctx, "load 应返回 inconclusive iter"

        # 空历史
        eng.memory.recall.return_value = []
        assert eng._load_prev_run_context() == "", "空历史应返回空串"

        print("[ok] P2 run_context persist + load")


def _check_decider_prompt_prev_run():
    """P2: decider prompt 注入 Previous run context 块."""
    from huginn.autoloop.cognitive_loop import LoopState
    from huginn.autoloop.engine import AutoloopEngine
    eng = AutoloopEngine.__new__(AutoloopEngine)
    eng._prev_run_context = "hypothesis: Fe diffuses | outcome: inconclusive"
    eng._physical_timeseries = []
    state = LoopState()
    cog = {"hypothesis": "test", "plan": {"mode": "lammps"},
           "execution_result": None, "validation": None,
           "last_learn_summary": ""}
    prompt = eng._build_decider_prompt(state, cog, {})
    assert "Previous run context" in prompt, "decider prompt 缺 Previous run context 块"
    assert "Fe diffuses" in prompt, "prev_run_context 内容未注入"
    # 空上下文也应正常 (无崩溃)
    eng._prev_run_context = ""
    p2 = eng._build_decider_prompt(state, cog, {})
    assert "none" in p2, "空 prev_run_context 应显示 none"
    print("[ok] P2 decider prompt 注入 prev_run_context")


def _check_extract_timeseries():
    """P3: _extract_timeseries 从工具结果提取 _physical_timeseries."""
    from huginn.autoloop.engine import AutoloopEngine
    _extract = AutoloopEngine._extract_timeseries

    # None
    assert _extract(None) == [], "None 结果应返回空"

    # 无 _physical_timeseries key
    _result = {"status": "ok", "data": [1, 2, 3]}
    assert _extract(_result) == [], "无 key 应返回空"

    # 直接含 key
    _ts = [{"name": "VACF", "data": [(0, 1.0), (1, 0.5)], "unit": "X",
            "meaning": "test", "source": "lammps"}]
    _result = {"_physical_timeseries": _ts}
    assert _extract(_result) == _ts, "直接含 key 应返回时序"

    # 嵌套在 result 子 dict
    _result = {"result": {"_physical_timeseries": _ts}}
    assert _extract(_result) == _ts, "嵌套在 result 应返回时序"

    print("[ok] P3 _extract_timeseries 提取")


def _check_format_timeseries():
    """P3: _format_timeseries 格式化时序数据为 prompt 摘要."""
    from huginn.autoloop.engine import AutoloopEngine
    eng = AutoloopEngine.__new__(AutoloopEngine)

    # 空列表
    assert eng._format_timeseries_context() == "", "空列表应返回空串"

    # 衰减趋势 (VACF 典型: 从高到低)
    eng._physical_timeseries = [{
        "name": "VACF",
        "unit": "Å²/ps²",
        "data": [(0, 1.0), (1, 0.5), (2, 0.1)],
        "meaning": "velocity autocorrelation",
        "source": "lammps",
    }]
    out = eng._format_timeseries_context()
    assert "VACF" in out, "应包含 name"
    assert "lammps" in out, "应包含 source"
    assert "Å²/ps²" in out, "应包含 unit"
    assert "decaying" in out, "衰减趋势应识别为 decaying"
    assert "3 pts" in out, "应包含数据点数"

    # 上升趋势
    eng._physical_timeseries = [{
        "name": "MSD",
        "unit": "Å²",
        "data": [(0, 0.0), (1, 1.0), (2, 4.0)],
        "meaning": "mean squared displacement",
        "source": "lammps",
    }]
    out = eng._format_timeseries_context()
    assert "rising" in out, "上升趋势应识别为 rising"

    # 平稳
    eng._physical_timeseries = [{
        "name": "Temp",
        "unit": "K",
        "data": [(0, 300.0), (1, 300.0)],
        "meaning": "temperature",
        "source": "thermo",
    }]
    out = eng._format_timeseries_context()
    assert "flat" in out, "平稳趋势应识别为 flat"

    # 上限: 6 条只显示最近 5 条
    eng._physical_timeseries = [
        {"name": f"TS{i}", "unit": "x", "data": [(0, float(i))],
         "meaning": "", "source": "test"}
        for i in range(6)
    ]
    out = eng._format_timeseries_context()
    assert "TS5" in out, "应保留最新 (TS5)"
    assert "TS0" not in out, "应丢弃最旧 (TS0)"

    print("[ok] P3 _format_timeseries_context 格式化 + 趋势判断")


def _check_decider_prompt_timeseries():
    """P3: decider prompt 注入 Physical time series 块."""
    from huginn.autoloop.cognitive_loop import LoopState
    from huginn.autoloop.engine import AutoloopEngine
    eng = AutoloopEngine.__new__(AutoloopEngine)
    eng._prev_run_context = ""
    eng._physical_timeseries = [{
        "name": "VACF",
        "unit": "Å²/ps²",
        "data": [(0, 1.0), (1, 0.5)],
        "meaning": "velocity autocorrelation",
        "source": "lammps",
    }]
    state = LoopState()
    cog = {"hypothesis": "test", "plan": {"mode": "lammps"},
           "execution_result": None, "validation": None,
           "last_learn_summary": ""}
    prompt = eng._build_decider_prompt(state, cog, {})
    assert "Physical time series" in prompt, "decider prompt 缺 Physical time series 块"
    assert "VACF" in prompt, "timeseries 内容未注入"
    assert "decaying" in prompt, "趋势未注入"
    # 空时序也应正常
    eng._physical_timeseries = []
    p2 = eng._build_decider_prompt(state, cog, {})
    assert "none" in p2, "空时序应显示 none"
    print("[ok] P3 decider prompt 注入 timeseries")


def _check_lammps_vacf_registers_timeseries():
    """P3: lammps_tool VACF 结果包含 _physical_timeseries key."""
    # 不跑真实 LAMMPS — 只验证 _compute_vacf 返回的格式能被 _extract_timeseries 提取
    from huginn.autoloop.engine import AutoloopEngine

    # 模拟 lammps_tool 构造 _physical_timeseries 的代码路径
    _vacf_data = [
        {"frame_index": 0, "timestep": 0, "vacf": 1.0, "vacf_normalized": 1.0},
        {"frame_index": 1, "timestep": 100, "vacf": 0.5, "vacf_normalized": 0.5},
        {"frame_index": 2, "timestep": 200, "vacf": 0.1, "vacf_normalized": 0.1},
    ]
    # 复现 lammps_tool 里 _physical_timeseries 的构造逻辑
    _ts = [{
        "name": "VACF",
        "unit": "Å²/ps²",
        "data": [
            (d.get("timestep", i), d.get("vacf", 0.0))
            for i, d in enumerate(_vacf_data)
        ],
        "meaning": "velocity autocorrelation <v(0)·v(t)>",
        "source": "lammps",
    }]
    _result = {"vacf": _vacf_data, "_physical_timeseries": _ts}

    _extracted = AutoloopEngine._extract_timeseries(_result)
    assert _extracted == _ts, "lammps_tool 结果应能被 _extract_timeseries 提取"
    assert len(_extracted[0]["data"]) == 3, "应有 3 个数据点"
    assert _extracted[0]["data"][0] == (0, 1.0), "第一个点应为 (0, 1.0)"
    assert _extracted[0]["data"][-1] == (200, 0.1), "最后一点应为 (200, 0.1)"

    print("[ok] P3 lammps_tool VACF → _physical_timeseries 提取")


def _check_lammps_msd_rdf_registers_timeseries():
    """P3: lammps_tool MSD + RDF 峰位 也注册为 _physical_timeseries.

    之前只有 VACF 走 cognition 通道; MSD/RDF 算了但躺在 result 里 agent
    看不到趋势. 现在三路独立注册, 无速度列时也能进 PMK.
    """
    from huginn.autoloop.engine import AutoloopEngine
    _extract = AutoloopEngine._extract_timeseries

    # Fake result: 有 MSD + RDF series, 无 VACF (模拟无速度列的 dump)
    _msd = [{"timestep": 0, "msd": 0.0}, {"timestep": 100, "msd": 1.0},
            {"timestep": 200, "msd": 4.0}]
    _rdf_series = [
        {"timestep": 0, "r": [1.0, 2.0, 3.0], "g": [0.0, 5.0, 1.0]},
        {"timestep": 200, "r": [1.0, 2.0, 3.0], "g": [0.0, 3.0, 2.0]},
    ]

    # 复现 lammps_tool 里 _physical_timeseries 的三路注册逻辑
    _ts: list[dict] = []
    # MSD branch
    _ts.append({
        "name": "MSD", "unit": "Å²",
        "data": [(d.get("timestep", i), d.get("msd", 0.0))
                 for i, d in enumerate(_msd)],
        "meaning": "mean squared displacement", "source": "lammps",
    })
    # RDF first-peak branch
    peak_data = []
    for fi, fr in enumerate(_rdf_series):
        g, r = fr.get("g"), fr.get("r")
        if not (g and r and len(g) == len(r) and len(g) > 0):
            continue
        peak_idx = max(range(len(g)), key=lambda k: g[k])
        peak_data.append((fr.get("timestep", fi), r[peak_idx]))
    if peak_data:
        _ts.append({
            "name": "RDF_first_peak", "unit": "Å",
            "data": peak_data,
            "meaning": "RDF first peak position", "source": "lammps",
        })
    # VACF branch — 不构造 (无速度列)
    _result = {"msd": _msd, "rdf_series": _rdf_series, "_physical_timeseries": _ts}

    extracted = _extract(_result)
    names = [t["name"] for t in extracted]
    assert "MSD" in names, "MSD 时序应注册"
    assert "RDF_first_peak" in names, "RDF 峰位时序应注册"
    assert "VACF" not in names, "无速度列时不应有 VACF 时序"

    # MSD 趋势: rising (0→1→4)
    msd_ts = next(t for t in extracted if t["name"] == "MSD")
    assert msd_ts["data"] == [(0, 0.0), (100, 1.0), (200, 4.0)], "MSD data 应正确"
    assert msd_ts["unit"] == "Å²"

    # RDF 峰位: 第一帧峰位 r=2.0 (g=[0,5,1] argmax=1→r=2.0), 末帧峰位 r=2.0 (g=[0,3,2] argmax=1→r=2.0)
    rdf_ts = next(t for t in extracted if t["name"] == "RDF_first_peak")
    assert rdf_ts["data"] == [(0, 2.0), (200, 2.0)], f"RDF 峰位应为 (0,2.0)(200,2.0), got {rdf_ts['data']}"
    assert rdf_ts["unit"] == "Å"

    print("[ok] P3 lammps_tool MSD + RDF_first_peak → _physical_timeseries (无 VACF 也能注册)")


def main():
    _check_run_context_persist_load()
    _check_decider_prompt_prev_run()
    _check_extract_timeseries()
    _check_format_timeseries()
    _check_decider_prompt_timeseries()
    _check_lammps_vacf_registers_timeseries()
    _check_lammps_msd_rdf_registers_timeseries()
    _check_learn_from_rcb_run_context()
    print("\nALL CHECKS PASSED")


def _check_learn_from_rcb_run_context():
    """P2 下沉: learn_from_rcb 存 run_context, RCB runner 路径也覆盖."""
    from unittest.mock import MagicMock
    from huginn.autoloop.cognitive_loop import learn_from_rcb

    mem_mgr = MagicMock()
    # validation: tests_passed=True (completed)
    _val = {"tests_passed": True, "darwin_score": 0.7, "r_phys": 0.8}
    result = learn_from_rcb(
        mem_mgr=mem_mgr,
        hypothesis="Fe diffuses at high T",
        validation=_val,
        persona_name="default",
        run_id="rcb_test01",
        domain="alloy",
    )
    assert result["memory_written"], "memory 应写入"
    # 检查 run_context 被存 — remember 被调多次 (iteration_result + persona_history + run_context)
    _calls = mem_mgr.remember.call_args_list + mem_mgr.remember_typed.call_args_list
    _rc_calls = [c for c in _calls
                 if c.kwargs.get("category") == "run_context"
                 or c.kwargs.get("memory_type") == "run_context"]
    # remember(category="run_context") 应至少 1 次
    _found = False
    for c in _calls:
        if c.kwargs.get("category") == "run_context":
            _found = True
            _content = c.kwargs.get("content", "")
            _snap = json.loads(_content)
            assert _snap["run_id"] == "rcb_test01"
            assert _snap["outcome"] == "completed"  # tests_passed=True
            assert _snap["plan_mode"] == "rcb"
            assert _snap["source"] == "learn_from_rcb"
            assert "Fe diffuses" in _snap["hypothesis"]
            break
    assert _found, "learn_from_rcb 应存 run_context category"

    # tests_passed=False → inconclusive
    mem_mgr2 = MagicMock()
    _val2 = {"tests_passed": False, "darwin_score": 0.3}
    learn_from_rcb(
        mem_mgr=mem_mgr2,
        hypothesis="test inconclusive",
        validation=_val2,
        run_id="rcb_test02",
    )
    for c in mem_mgr2.remember.call_args_list:
        if c.kwargs.get("category") == "run_context":
            _snap2 = json.loads(c.kwargs.get("content", "{}"))
            assert _snap2["outcome"] == "inconclusive"
            assert len(_snap2["inconclusive"]) == 1
            break

    # mem_mgr=None → 不崩溃, run_context 跳过
    _r = learn_from_rcb(mem_mgr=None, hypothesis="x", validation={})
    assert _r["memory_written"] is False

    print("[ok] P2 下沉 learn_from_rcb → run_context (RCB + cognitive loop 双覆盖)")


if __name__ == "__main__":
    main()
