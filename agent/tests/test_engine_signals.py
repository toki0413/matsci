"""EngineSignals 收敛重构的 TDD 测试:
 1) snapshot round-trip(含 set/tuple 嵌套)
 2) 真实 AutoloopEngine 的属性桥 self._x <-> self.signals.x
 3) engine_state 持久化：环信号(经 signals) + 基建态都 round-trip
"""

from __future__ import annotations

from huginn.autoloop.signals import EngineSignals


def test_signals_snapshot_roundtrip():
    s = EngineSignals()
    s._iteration = 7
    s._last_surprise = 0.42
    s._validate_window = [True, False, True]
    s._scene_tag_extra_keywords = {"dft": {"iso", "relax"}}
    s._surprise_history = [(0.1, 0.2), (0.3, 0.4)]
    s._darwin_belief_sigma2 = 101.0
    s._refined_hypothesis = "h_new"

    snap = s.to_snapshot()
    # JSON 可序列化：不应残留 set/tuple
    import json

    json.loads(json.dumps(snap))

    s2 = EngineSignals.from_snapshot(snap)
    assert s2._iteration == 7
    assert s2._last_surprise == 0.42
    assert s2._validate_window == [True, False, True]
    assert s2._scene_tag_extra_keywords == {"dft": {"iso", "relax"}}
    assert s2._surprise_history == [(0.1, 0.2), (0.3, 0.4)]
    assert s2._darwin_belief_sigma2 == 101.0
    assert s2._refined_hypothesis == "h_new"


def test_from_snapshot_missing_keys_use_defaults():
    s = EngineSignals.from_snapshot({"_iteration": 3})
    assert s._iteration == 3
    assert s._last_surprise == 0.0  # 默认
    assert s._surprise_history == []  # 默认
    assert s._scene_tag_extra_keywords == {}


def test_property_bridge_preserves_reads():
    from huginn.autoloop.engine import AutoloopEngine

    # __new__ 跳过 __init__，只验证属性桥 self._x <-> self.signals.x
    eng = AutoloopEngine.__new__(AutoloopEngine)
    eng.signals = EngineSignals()
    eng.signals._iteration = 5
    eng.signals._last_surprise = 0.9
    eng.signals._next_phase_hint = "execute"
    eng.signals._validate_window = [True]

    assert eng._iteration == 5
    assert eng._last_surprise == 0.9
    assert eng._next_phase_hint == "execute"
    assert eng._validate_window == [True]

    # 写也有桥
    eng._iteration = 42
    eng._next_phase_hint = "plan"
    assert eng.signals._iteration == 42
    assert eng.signals._next_phase_hint == "plan"


def test_engine_state_signals_persist(tmp_path):
    import os

    from huginn.runtime import engine_state as es

    prev = os.environ.get(es._PERSISTENCE_FLAG)
    os.environ[es._PERSISTENCE_FLAG] = "1"
    ws = tmp_path / "ws"
    ws.mkdir()

    class _FakeEngine:
        def __init__(self):
            self.signals = EngineSignals()
            self.signals._iteration = 7
            self.signals._consecutive_failures = 3
            self.signals._next_phase_hint = "execute"
            self.signals._surprise_history = [(0.2, 0.1)]
            self.hypothesis_graph = None
            # 基建态照旧在 engine 上
            self._budget_rejects = {"t1": 2}
            self._budget_degraded = True
            self._last_persona = "domain_skeptic"

    eng = _FakeEngine()
    try:
        saved = es.save_engine_state(eng, "run_sig", ws)
        assert saved is not None
        loaded = es.load_engine_state("run_sig", ws)
        assert loaded is not None
        # 环信号经 signals 槽恢复
        assert loaded.signals["_iteration"] == 7
        assert loaded.signals["_consecutive_failures"] == 3
        assert loaded.signals["_next_phase_hint"] == "execute"
        # 基建态照旧
        assert loaded._budget_rejects == {"t1": 2}
        assert loaded._budget_degraded is True
        assert loaded._last_persona == "domain_skeptic"

        eng2 = _FakeEngine()
        es.apply_state_to_engine(loaded, eng2)
        assert eng2.signals._iteration == 7
        assert eng2.signals._consecutive_failures == 3
        assert eng2.signals._next_phase_hint == "execute"
        assert eng2._budget_degraded is True
    finally:
        if prev is None:
            os.environ.pop(es._PERSISTENCE_FLAG, None)
        else:
            os.environ[es._PERSISTENCE_FLAG] = prev
