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
        "huginn.autoloop.engine.MemoryManager", lambda: MagicMock()
    )
    monkeypatch.setattr(
        "huginn.autoloop.engine.ProjectKnowledgeGraph", lambda: MagicMock()
    )
    monkeypatch.setattr("huginn.autoloop.engine.BenchmarkRunner", lambda: MagicMock())
    monkeypatch.setattr("huginn.autoloop.engine.CoderRunner", lambda: MagicMock())
    # speculator's on_turn_start does scenario matching; skip it
    monkeypatch.setattr(
        "huginn.agents.speculator.on_turn_start",
        lambda *a, **kw: {"hint": "", "predictions": []},
        raising=False,
    )
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
        result = asyncio.run(engine.run(objective="o", max_iterations=1))
        assert isinstance(result, AutoloopResult)

    def test_field_types(self, engine: AutoloopEngine):
        _patch_all_phases(engine)
        result = asyncio.run(engine.run(objective="my obj", max_iterations=1))
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
        result = asyncio.run(engine.run(objective="o", max_iterations=1))
        names = [p.name for p in result.phases]
        # 6 stages of one iteration + final report
        assert names == [
            "perceive",
            "hypothesize",
            "plan",
            "execute",
            "validate",
            "learn",
            "report",
        ]

    def test_phase_has_status_and_timestamps(self, engine: AutoloopEngine):
        _patch_all_phases(engine)
        result = asyncio.run(engine.run(objective="o", max_iterations=1))
        for phase in result.phases:
            assert phase.name
            assert phase.status in ("completed", "failed", "running", "pending")
            assert phase.start_time is not None
            assert phase.end_time is not None
            assert phase.end_time >= phase.start_time

    def test_success_true_when_all_phases_completed(self, engine: AutoloopEngine):
        _patch_all_phases(engine)
        result = asyncio.run(engine.run(objective="o", max_iterations=1))
        # success = all of last 7 phases completed
        assert result.success is True

    def test_report_path_propagated(self, engine: AutoloopEngine):
        _patch_all_phases(engine)
        expected = str(engine.workspace / "my_report.md")
        engine._report = AsyncMock(return_value=expected)  # type: ignore[assignment]
        result = asyncio.run(engine.run(objective="o", max_iterations=1))
        assert result.report_path == expected


class TestMaxIterations:
    def test_two_iterations_double_the_inner_phases(self, engine: AutoloopEngine):
        _patch_all_phases(engine)
        result = asyncio.run(engine.run(objective="o", max_iterations=2))
        # 2 iterations * 6 phases + 1 report = 13
        assert len(result.phases) == 13
        # first 6 names are iteration 1, next 6 are iteration 2, last is report
        names = [p.name for p in result.phases]
        assert names.count("perceive") == 2
        assert names.count("report") == 1

    def test_zero_iterations_only_report(self, engine: AutoloopEngine):
        _patch_all_phases(engine)
        result = asyncio.run(engine.run(objective="o", max_iterations=0))
        # while loop body never executes; only the post-loop report runs
        assert len(result.phases) == 1
        assert result.phases[0].name == "report"


class TestStop:
    def test_stop_attribute_exists_and_sets_flag(self, engine: AutoloopEngine):
        assert engine._should_stop is False
        engine.stop()
        assert engine._should_stop is True

    def test_stop_honoured_mid_run(self, engine: AutoloopEngine):
        # perceive calls stop() then returns a context — current iteration
        # still finishes (6 phases) but the while guard blocks iteration 2.
        def perceive_with_stop():
            engine.stop()
            return {"changed_files": ["x.py"], "timestamp": "t"}

        engine._perceive = perceive_with_stop  # type: ignore[assignment]
        _patch_all_phases(engine)
        engine._perceive = perceive_with_stop  # type: ignore[assignment]
        result = asyncio.run(engine.run(objective="o", max_iterations=5))
        # only one iteration ran (6 phases) + report = 7 total
        assert len(result.phases) == 7
        assert [p.name for p in result.phases].count("perceive") == 1


class TestPerceiveCallableDirectly:
    """Watch mode (autoloop.py:157) calls engine._perceive() without run().
    P1 refactor must keep this method callable."""

    def test_perceive_returns_dict_or_none(self, engine: AutoloopEngine):
        # Default _perceive uses PerceptionLayer; in an empty tmp_path it
        # should return None (no activity) without raising.
        result = engine._perceive()
        assert result is None or isinstance(result, dict)
