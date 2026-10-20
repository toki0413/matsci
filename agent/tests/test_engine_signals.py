"""EngineSignals 收敛重构的 TDD 测试:
 1) snapshot round-trip(含 set/tuple 嵌套)
 2) 真实 AutoloopEngine 的属性桥 self._x <-> self.signals.x
 3) engine_state 持久化：环信号(经 signals) + 基建态都 round-trip
"""

from __future__ import annotations

from huginn.autoloop.signals import (
    EngineSignals,
    hypothesis_strength,
    strength_branch_depth,
    strength_global_proposal_prob,
    strength_schedule_enabled,
    strength_stagnation_limit,
    strength_temperature,
)


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


# ── Ataraxos 式强度调度 (explore 超参自适应) ─────────────────────────


class _StubEngine:
    """最小 stub: 只提供调度读的纯环字段."""

    def __init__(self, **kw):
        self._last_surprise_rel = kw.get("rel")
        self._last_surprise = kw.get("raw", 0.0)
        self._validate_window = kw.get("window", [])
        self._darwin_best_score = kw.get("best", 0.0)
        self._darwin_stagnation = kw.get("stag", 0)


def test_strength_schedules_anchor_at_old_defaults():
    """s=0.5 处各调度必须回到旧硬编码默认 → 开/关平滑."""
    assert abs(strength_global_proposal_prob(0.5) - 0.3) < 1e-9
    assert abs(strength_temperature(0.5) - 1.0) < 1e-9
    assert strength_branch_depth(0.5) == 2
    assert strength_stagnation_limit(0.5) == 5


def test_strength_monotonic_weak_explores_strong_converges():
    # 弱 (s=0): 多全局跳 / 高温 / 深搜 / 早 pivot
    assert strength_global_proposal_prob(0.0) > strength_global_proposal_prob(1.0)
    assert strength_temperature(0.0) > strength_temperature(1.0)
    assert strength_branch_depth(0.0) > strength_branch_depth(1.0)
    assert strength_stagnation_limit(0.0) < strength_stagnation_limit(1.0)
    # 范围不越界
    assert strength_branch_depth(1.0) >= 1
    assert strength_stagnation_limit(1.0) <= 9


def test_hypothesis_strength_reads_signals_and_is_neutral_on_empty():
    weak = _StubEngine(rel=0.9, window=[False, False, True], best=1.0, stag=2)
    strong = _StubEngine(rel=0.1, window=[True] * 5, best=9.0, stag=0)
    assert hypothesis_strength(weak) < hypothesis_strength(strong)
    # 无 history (rel 缺 + raw 0) → 中性偏探索, 不误判为强
    empty = _StubEngine()
    assert 0.35 < hypothesis_strength(empty) < 0.6


def test_hypothesis_strength_tolerates_missing_fields():
    class _Bare:
        pass

    s = hypothesis_strength(_Bare())
    assert 0.0 <= s <= 1.0


def test_hypothesis_strength_observer_disagreement_lowers_strength():
    """受控独立观察者: 本轮观测到分歧 → 降 strength (转探索); 无观测零扰动."""
    kw = {"rel": 0.3, "window": [True] * 5, "best": 6.0, "stag": 0}
    neutral = hypothesis_strength(_StubEngine(**kw))
    eng = _StubEngine(**kw)
    # 无观测 / 一致 (None/False) → 与无观察者完全一致
    eng._last_reconstruct_disagree = None
    assert hypothesis_strength(eng) == neutral
    eng._last_reconstruct_disagree = False
    assert hypothesis_strength(eng) == neutral
    # 分歧 + 高置信 → 精确降 0.1
    eng._last_reconstruct_disagree = True
    eng._last_blind_confidence = 1.0
    assert abs(hypothesis_strength(eng) - (neutral - 0.1)) < 1e-9
    # 分歧 + 零置信 → 扰动为 0 (不轻信无把握的观察者)
    eng._last_blind_confidence = 0.0
    assert abs(hypothesis_strength(eng) - neutral) < 1e-9


def test_strength_schedule_flag_default_on_and_can_disable(monkeypatch):
    monkeypatch.delenv("HUGINN_STRENGTH_SCHEDULE", raising=False)
    assert strength_schedule_enabled() is True
    monkeypatch.setenv("HUGINN_STRENGTH_SCHEDULE", "0")
    assert strength_schedule_enabled() is False
