"""H3: autoloop engine 事件溯源 — phase 事件 → AutoloopStateProjection 投影.

验证:
  * AutoloopStateProjection 从 autoloop_phase_change 事件折叠出
    {phase, status, iteration, phase_seq} 读模型.
  * 冷重建 (从磁盘重放) == 热驱动, 投影可重放.
  * 非 autoloop 事件 (message 等) 不影响投影.
  * engine._record_autoloop_phase + read_runtime_state() 闭环:
    写事件日志 → 从投影读, 而非只看可变属性.
  * 无事件日志时 read_runtime_state() 回退内存属性 (向后兼容).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from huginn.autoloop.engine import AutoloopEngine
from huginn.events.projection import AutoloopStateProjection, ProjectionEngine
from huginn.events.session_log import (
    EVENT_AUTOLOOP_PHASE,
    EVENT_MESSAGE,
    SessionEventLog,
)


@pytest.fixture
def engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AutoloopEngine:
    """Build an engine with heavy sub-components stubbed (见 test_autoloop_engine)."""
    monkeypatch.setattr(
        "huginn.autoloop.engine.get_model", lambda settings: MagicMock()
    )
    monkeypatch.setattr(
        "huginn.autoloop.engine.MemoryManager", lambda *a, **kw: MagicMock()
    )
    monkeypatch.setattr(
        "huginn.autoloop.engine.ProjectKnowledgeGraph", lambda *a, **kw: MagicMock()
    )
    monkeypatch.setattr(
        "huginn.autoloop.engine.BenchmarkRunner", lambda *a, **kw: MagicMock()
    )
    monkeypatch.setattr(
        "huginn.autoloop.engine.CoderRunner", lambda *a, **kw: MagicMock()
    )
    monkeypatch.setattr(
        "huginn.autoloop.engine.AutoloopEngine._get_kb", lambda self: None
    )
    monkeypatch.setattr(
        "huginn.autoloop.conjecture.get_kg", lambda *a, **kw: None
    )
    return AutoloopEngine(workspace=tmp_path)


# ── projection 纯函数单元 ────────────────────────────────────────────
def _drive(log: SessionEventLog, engine: ProjectionEngine) -> None:
    for ev in log:
        engine.drive(log, ev)


def test_autoloop_projection_folds_phase_events(tmp_path: Path) -> None:
    log = SessionEventLog("s1", Path(tmp_path) / "s1.jsonl", load=False)
    eng = ProjectionEngine()
    eng.register(AutoloopStateProjection())

    log.append(EVENT_AUTOLOOP_PHASE, {"phase": "perceive", "status": "running", "iteration": 0})
    log.append(EVENT_AUTOLOOP_PHASE, {"phase": "hypothesize", "status": "running", "iteration": 0})
    log.append(EVENT_AUTOLOOP_PHASE, {"phase": "hypothesize", "status": "completed", "iteration": 0})
    log.append(EVENT_AUTOLOOP_PHASE, {"phase": "report", "status": "completed", "iteration": 3})
    _drive(log, eng)

    state = eng.build(log, "autoloop")
    assert state["phase"] == "report"
    assert state["status"] == "completed"
    assert state["iteration"] == 3
    # phase_seq 是最新 phase 事件的 seq (0-based)
    assert state["phase_seq"] == 3


def test_autoloop_projection_cold_rebuild_matches(tmp_path: Path) -> None:
    log = SessionEventLog("s1", Path(tmp_path) / "s1.jsonl", load=False)
    log.append(EVENT_AUTOLOOP_PHASE, {"phase": "plan", "status": "running", "iteration": 1})
    log.append(EVENT_AUTOLOOP_PHASE, {"phase": "execute", "status": "running", "iteration": 1})

    eng1 = ProjectionEngine()
    eng1.register(AutoloopStateProjection())
    driven = eng1.build(log, "autoloop")

    # 冷引擎从磁盘重放 → 必须等于热驱动结果 (事件日志 = source of truth)
    log2 = SessionEventLog("s1", Path(tmp_path) / "s1.jsonl", load=True)
    eng2 = ProjectionEngine()
    eng2.register(AutoloopStateProjection())
    assert eng2.build(log2, "autoloop") == driven


def test_autoloop_projection_ignores_other_kinds(tmp_path: Path) -> None:
    log = SessionEventLog("s1", Path(tmp_path) / "s1.jsonl", load=False)
    eng = ProjectionEngine()
    eng.register(AutoloopStateProjection())

    log.append(EVENT_MESSAGE, {"role": "user", "content": "hi"})
    log.append(EVENT_AUTOLOOP_PHASE, {"phase": "perceive", "status": "running", "iteration": 0})
    log.append(EVENT_MESSAGE, {"role": "assistant", "content": "ok"})
    _drive(log, eng)

    state = eng.build(log, "autoloop")
    assert state["phase"] == "perceive"
    # 两条 message 事件不影响 phase 折叠
    assert state["phase_seq"] == 1


# ── engine 闭环 ──────────────────────────────────────────────────────
def test_engine_records_and_reads_runtime_state(engine: AutoloopEngine) -> None:
    engine._record_autoloop_phase("perceive", "running", 0)
    engine._record_autoloop_phase("hypothesize", "running", 0)
    engine._record_autoloop_phase("hypothesize", "completed", 0)
    engine._record_autoloop_phase("report", "completed", 3)

    state = engine.read_runtime_state()
    # 从事件投影读 → 与日志强一致, 而非可变快照
    assert state["phase"] == "report"
    assert state["status"] == "completed"
    assert state["iteration"] == 3

    # 事件日志落盘存在
    assert engine._event_log_path().exists()


def test_engine_runtime_state_replays_from_disk(engine: AutoloopEngine) -> None:
    engine._record_autoloop_phase("plan", "running", 1)
    engine._record_autoloop_phase("execute", "running", 1)

    # 新引擎 (同 workspace) 从磁盘重放事件 → 恢复出同一 phase 位置
    engine2 = AutoloopEngine(workspace=engine.workspace)
    state2 = engine2.read_runtime_state()
    assert state2["phase"] == "execute"
    assert state2["iteration"] == 1


def test_engine_runtime_state_fallback_without_log(engine: AutoloopEngine) -> None:
    # 不写任何事件 → 日志为空 → 回退内存属性 (向后兼容)
    engine._current_phase = "validate"
    engine._iteration = 5
    state = engine.read_runtime_state()
    assert state["phase"] == "validate"
    assert state["iteration"] == 5
    assert state["phase_seq"] == -1


# ── H3 UI/进度通道: 投影 → ProgressTracker ──────────────────────────
class _RecordingTracker:
    """Record tracker.update calls for assertions."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def update(self, task_id: str, **kw) -> None:
        self.calls.append({"task_id": task_id, **kw})

    def start_task(self, *a, **kw) -> None: ...
    def complete(self, *a, **kw) -> None: ...
    def fail(self, *a, **kw) -> None: ...


def test_publish_progress_pushes_projection_to_tracker(engine: AutoloopEngine) -> None:
    engine._record_autoloop_phase("perceive", "running", 0)
    engine._record_autoloop_phase("hypothesize", "completed", 2)
    engine._progress_task_id = "autoloop:abc"
    tracker = _RecordingTracker()
    engine.progress_tracker = tracker

    engine._publish_progress()

    assert len(tracker.calls) == 1
    call = tracker.calls[0]
    assert call["task_id"] == "autoloop:abc"
    # current_label 来自投影读模型
    assert call["current_label"] == "hypothesize (completed)"
    meta = call["metadata"]
    assert meta["phase"] == "hypothesize"
    assert meta["phase_status"] == "completed"
    assert meta["iteration"] == 2
    assert meta["source"] == "event_projection"
    # phase_seq 携带最新 phase 事件 seq (0-based)
    assert meta["phase_seq"] == 1


def test_publish_progress_noop_without_task_id(engine: AutoloopEngine) -> None:
    engine._record_autoloop_phase("perceive", "running", 0)
    tracker = _RecordingTracker()
    engine.progress_tracker = tracker
    # 未设置 _progress_task_id → 不推送, 也不抛错
    engine._progress_task_id = None
    engine._publish_progress()
    assert tracker.calls == []


def test_publish_progress_falls_back_to_memory_state(engine: AutoloopEngine) -> None:
    # 无事件日志时, 投影回退内存属性, 进度通道仍能推送
    engine._current_phase = "validate"
    engine._iteration = 3
    engine._progress_task_id = "autoloop:def"
    tracker = _RecordingTracker()
    engine.progress_tracker = tracker

    engine._publish_progress()

    assert len(tracker.calls) == 1
    meta = tracker.calls[0]["metadata"]
    assert meta["phase"] == "validate"
    assert meta["iteration"] == 3
    assert meta["phase_seq"] == -1
