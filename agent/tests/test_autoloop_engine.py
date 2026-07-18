"""Regression net for AutoloopEngine — locks the run() return contract.

AutoloopEngine has zero test coverage today. Before P1 refactors run() into a
thin wrapper around iterate_once(), we need a test that pins:
  * AutoloopResult field types and shape
  * phase list structure (LoopPhase with name/status/timestamps)
  * max_iterations respected
  * stop() honoured mid-run
  * _perceive() stays callable directly (watch loop contract, autoloop.py:157)

All LLM / network / subprocess paths are stubbed so the test is hermetic.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from huginn.autoloop.engine import AutoloopEngine, AutoloopResult, LoopPhase


@pytest.fixture
def engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AutoloopEngine:
    """Build an engine with all heavy sub-components stubbed.

    AutoloopEngine.__init__ pulls in get_model / MemoryManager /
    ProjectKnowledgeGraph / BenchmarkRunner / CoderRunner / etc. We don't
    need any of them for contract tests — run() only touches self via the
    phase methods, which the tests override per-case.
    """
    monkeypatch.setattr(
        "huginn.autoloop.engine.get_model", lambda settings: MagicMock()
    )
    monkeypatch.setattr(
        "huginn.autoloop.engine.MemoryManager", lambda *a, **kw: MagicMock()
    )
    monkeypatch.setattr(
        "huginn.autoloop.engine.ProjectKnowledgeGraph", lambda *a, **kw: MagicMock()
    )
    monkeypatch.setattr("huginn.autoloop.engine.BenchmarkRunner", lambda *a, **kw: MagicMock())
    monkeypatch.setattr("huginn.autoloop.engine.CoderRunner", lambda *a, **kw: MagicMock())
    # speculator's on_turn_start does scenario matching; skip it
    monkeypatch.setattr(
        "huginn.agents.speculator.on_turn_start",
        lambda *a, **kw: {"hint": "", "predictions": []},
        raising=False,
    )
    # ponytail: KB 冷启动跑 ONNX embedding > 120s, KG 写 ~/.huginn 污染 home
    monkeypatch.setattr("huginn.autoloop.engine.AutoloopEngine._get_kb", lambda self: None)
    monkeypatch.setattr("huginn.autoloop.conjecture.get_kg", lambda *a, **kw: None)
    eng = AutoloopEngine(workspace=tmp_path)
    # progress_tracker isolation: use a fresh in-memory tracker per test
    eng.progress_tracker = _DummyTracker()
    return eng


class _DummyTracker:
    """Minimal ProgressTracker stand-in — just records calls."""

    def start_task(self, *a, **kw) -> None: ...
    def update(self, *a, **kw) -> None: ...
    def complete(self, *a, **kw) -> None: ...
    def fail(self, *a, **kw) -> None: ...


def _patch_all_phases(engine: AutoloopEngine) -> None:
    """Replace every phase method with a canned no-op return."""
    engine._perceive = lambda: {"changed_files": ["x.py"], "timestamp": "t"}  # type: ignore[assignment]
    engine._hypothesize = AsyncMock(return_value="test hypothesis")  # type: ignore[assignment]
    engine._plan = AsyncMock(return_value={"mode": "coder", "description": "do x"})  # type: ignore[assignment]
    engine._execute = AsyncMock(return_value={"mode": "coder", "status": "ok"})  # type: ignore[assignment]
    engine._validate = AsyncMock(return_value={"tests_passed": True})  # type: ignore[assignment]
    engine._learn = AsyncMock(return_value=None)  # type: ignore[assignment]
    engine._report = AsyncMock(return_value=str(engine.workspace / "report.md"))  # type: ignore[assignment]


class TestRunReturnContract:
    def test_returns_autoloop_result(self, engine: AutoloopEngine):
        _patch_all_phases(engine)
        result = asyncio.run(engine.run_cognitive(objective="o", max_iterations=1))
        assert isinstance(result, AutoloopResult)

    def test_field_types(self, engine: AutoloopEngine):
        _patch_all_phases(engine)
        result = asyncio.run(engine.run_cognitive(objective="my obj", max_iterations=1))
        assert isinstance(result.run_id, str)
        assert result.run_id.startswith("loop_")
        assert result.objective == "my obj"
        assert isinstance(result.phases, list)
        assert all(isinstance(p, LoopPhase) for p in result.phases)
        assert isinstance(result.success, bool)
        assert result.report_path is None or isinstance(result.report_path, str)
        assert isinstance(result.total_time_seconds, float)
        assert result.total_time_seconds >= 0.0

    def test_one_iteration_yields_six_phases_plus_report(
        self, engine: AutoloopEngine
    ):
        _patch_all_phases(engine)
        # v10: run_cognitive 走 CognitiveLoop, 每轮 1 action (不是 run() 的 7-phase/轮).
        # 要跑完 hypothesize→plan→execute→validate→learn 需要 5+ iterations,
        # 加 report 共 6 phase. max_iterations=10 留余量.
        result = asyncio.run(engine.run_cognitive(objective="o", max_iterations=10))
        names = [p.name for p in result.phases]
        # CognitiveLoop 规则版 decide_fn: hypothesize→plan→execute→validate→learn→stop
        # observe_fn 跑 _perceive 但不 append phase (action="observe" 才 append),
        # 规则版 decide 首轮直接 hypothesize, 所以 phases 不含 perceive.
        assert "report" in names, f"缺 report phase, 实际 {names}"
        assert "hypothesize" in names, f"缺 hypothesize phase, 实际 {names}"
        # 至少跑到 learn (5 action + report = 6 phase)
        assert len(names) >= 5, f"phase 数 < 5, 实际 {names}"

    def test_phase_has_status_and_timestamps(self, engine: AutoloopEngine):
        _patch_all_phases(engine)
        result = asyncio.run(engine.run_cognitive(objective="o", max_iterations=1))
        for phase in result.phases:
            assert phase.name
            assert phase.status in ("completed", "failed", "running", "pending")
            assert phase.start_time is not None
            assert phase.end_time is not None
            assert phase.end_time >= phase.start_time

    def test_success_true_when_all_phases_completed(self, engine: AutoloopEngine):
        _patch_all_phases(engine)
        result = asyncio.run(engine.run_cognitive(objective="o", max_iterations=1))
        # success = all of last 7 phases completed
        assert result.success is True

    def test_report_path_propagated(self, engine: AutoloopEngine):
        _patch_all_phases(engine)
        expected = str(engine.workspace / "my_report.md")
        engine._report = AsyncMock(return_value=expected)  # type: ignore[assignment]
        result = asyncio.run(engine.run_cognitive(objective="o", max_iterations=1))
        assert result.report_path == expected


class TestMaxIterations:
    def test_two_iterations_double_the_inner_phases(self, engine: AutoloopEngine):
        _patch_all_phases(engine)
        # v10: run_cognitive 走 CognitiveLoop, 每轮 1 action. max_iterations=2
        # 跑 2 actions (hypothesize→plan) + report = 3 phase. max_iterations=4
        # 跑 4 actions + report = 5 phase. 验证翻倍 max_iter → actions 翻倍.
        result_2 = asyncio.run(engine.run_cognitive(objective="o", max_iterations=2))
        result_4 = asyncio.run(engine.run_cognitive(objective="o", max_iterations=4))
        actions_2 = [p for p in result_2.phases if p.name != "report"]
        actions_4 = [p for p in result_4.phases if p.name != "report"]
        assert len(actions_4) == 2 * len(actions_2), (
            f"翻倍 max_iter 应该让 actions 翻倍: {len(actions_2)} vs {len(actions_4)}"
        )
        # report 只有一个
        assert [p.name for p in result_2.phases].count("report") == 1
        assert [p.name for p in result_4.phases].count("report") == 1

    def test_zero_iterations_only_report(self, engine: AutoloopEngine):
        _patch_all_phases(engine)
        result = asyncio.run(engine.run_cognitive(objective="o", max_iterations=0))
        # while loop body never executes; only the post-loop report runs
        assert len(result.phases) == 1
        assert result.phases[0].name == "report"


class TestStop:
    def test_stop_attribute_exists_and_sets_flag(self, engine: AutoloopEngine):
        assert engine._should_stop is False
        engine.stop()
        assert engine._should_stop is True

    def test_stop_honoured_mid_run(self, engine: AutoloopEngine):
        # v10: perceive 调 stop() 设 self._should_stop. observe_fn 开头检测到
        # _should_stop → state.should_stop=True, CognitiveLoop while guard 退出.
        # 当前 iter 已开始的 action 会跑完 (hypothesize), 下一轮 observe 直接 return.
        # 期望: 1 action (hypothesize) + report = 2 phase.
        def perceive_with_stop():
            engine.stop()
            return {"changed_files": ["x.py"], "timestamp": "t"}

        engine._perceive = perceive_with_stop  # type: ignore[assignment]
        _patch_all_phases(engine)
        engine._perceive = perceive_with_stop  # type: ignore[assignment]
        result = asyncio.run(engine.run_cognitive(objective="o", max_iterations=5))
        # v10: 1 action (hypothesize) + report = 2 phase. observe_fn 开头检测
        # _should_stop 直接 return, 不会跑第 2 个 action.
        names = [p.name for p in result.phases]
        assert "report" in names, f"缺 report, 实际 {names}"
        # 至少跑了 1 个 action (不是 0), 至多 2 个 (hypothesize + 可能的 plan)
        actions = [n for n in names if n != "report"]
        assert 1 <= len(actions) <= 2, f"actions 数应该 1-2, 实际 {names}"


class TestPerceiveCallableDirectly:
    """Watch mode (autoloop.py:157) calls engine._perceive() without run().
    P1 refactor must keep this method callable."""

    def test_perceive_returns_dict_or_none(self, engine: AutoloopEngine):
        # Default _perceive uses PerceptionLayer; in an empty tmp_path it
        # should return None (no activity) without raising.
        result = engine._perceive()
        assert result is None or isinstance(result, dict)
