"""Tests for GoalScheduler + Goal (W2 A4).

Covers:
- Goal dataclass defaults + timestamp auto-fill
- GoalScheduler CRUD: create / get / list / update / activate / complete / fail / delete
- Persistence: save→load roundtrip, new instance reloads from disk, corrupt file survives
- check_completion: keyword matching, case-insensitive, multiple criteria, edge cases
- Engine integration: run(goal=...) stops early when criteria met, no goal = no change
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from huginn.autoloop.engine import AutoloopEngine
from huginn.autoloop.goal_scheduler import Goal, GoalScheduler


# ── Goal dataclass ───────────────────────────────────────────────────────────


class TestGoalDataclass:
    def test_defaults_auto_fill_timestamps(self):
        g = Goal(id="g1", objective="obj", success_criteria=["a"])
        assert g.status == "pending"
        assert g.max_iterations == 20
        assert g.created_at  # auto-filled ISO string
        assert g.updated_at == g.created_at
        assert g.completed_at is None

    def test_explicit_timestamps_respected(self):
        g = Goal(
            id="g2",
            objective="obj",
            success_criteria=[],
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-02T00:00:00+00:00",
        )
        assert g.created_at == "2026-01-01T00:00:00+00:00"
        assert g.updated_at == "2026-01-02T00:00:00+00:00"

    def test_metadata_defaults_to_empty_dict(self):
        g = Goal(id="g3", objective="o", success_criteria=[])
        assert g.metadata == {}


# ── GoalScheduler CRUD ───────────────────────────────────────────────────────


class TestSchedulerCRUD:
    def test_create_goal_assigns_id_and_persists(self, tmp_path):
        sched = GoalScheduler(path=tmp_path / "goals.json")
        g = sched.create_goal("optimize", success_criteria=["r_phys"])
        assert g.id.startswith("goal_")
        assert g.objective == "optimize"
        assert g.success_criteria == ["r_phys"]
        assert g.status == "pending"
        # persisted to disk
        assert (tmp_path / "goals.json").exists()

    def test_get_goal_returns_none_for_missing(self, tmp_path):
        sched = GoalScheduler(path=tmp_path / "goals.json")
        assert sched.get_goal("nonexistent") is None

    def test_get_goal_returns_created(self, tmp_path):
        sched = GoalScheduler(path=tmp_path / "goals.json")
        g = sched.create_goal("obj", success_criteria=["x"])
        assert sched.get_goal(g.id) is g

    def test_list_goals_filters_by_status(self, tmp_path):
        sched = GoalScheduler(path=tmp_path / "goals.json")
        g1 = sched.create_goal("a", success_criteria=[])
        g2 = sched.create_goal("b", success_criteria=[])
        sched.complete_goal(g1.id)
        active = sched.list_goals(status="completed")
        assert len(active) == 1
        assert active[0].id == g1.id
        all_goals = sched.list_goals()
        assert len(all_goals) == 2

    def test_update_goal_changes_fields(self, tmp_path):
        sched = GoalScheduler(path=tmp_path / "goals.json")
        g = sched.create_goal("obj", success_criteria=["x"])
        updated = sched.update_goal(g.id, objective="new obj", max_iterations=10)
        assert updated.objective == "new obj"
        assert updated.max_iterations == 10
        assert updated.updated_at >= g.updated_at

    def test_update_goal_raises_for_missing(self, tmp_path):
        sched = GoalScheduler(path=tmp_path / "goals.json")
        with pytest.raises(KeyError):
            sched.update_goal("nope", objective="x")

    def test_activate_goal(self, tmp_path):
        sched = GoalScheduler(path=tmp_path / "goals.json")
        g = sched.create_goal("obj", success_criteria=[])
        assert g.status == "pending"
        sched.activate_goal(g.id)
        assert sched.get_goal(g.id).status == "active"

    def test_complete_goal_sets_completed_at(self, tmp_path):
        sched = GoalScheduler(path=tmp_path / "goals.json")
        g = sched.create_goal("obj", success_criteria=[])
        sched.complete_goal(g.id)
        completed = sched.get_goal(g.id)
        assert completed.status == "completed"
        assert completed.completed_at is not None

    def test_fail_goal_records_reason(self, tmp_path):
        sched = GoalScheduler(path=tmp_path / "goals.json")
        g = sched.create_goal("obj", success_criteria=[])
        sched.fail_goal(g.id, reason="budget exhausted")
        failed = sched.get_goal(g.id)
        assert failed.status == "failed"
        assert failed.metadata.get("failure_reason") == "budget exhausted"

    def test_delete_goal(self, tmp_path):
        sched = GoalScheduler(path=tmp_path / "goals.json")
        g = sched.create_goal("obj", success_criteria=[])
        assert sched.delete_goal(g.id) is True
        assert sched.get_goal(g.id) is None
        assert sched.delete_goal(g.id) is False


# ── Persistence ──────────────────────────────────────────────────────────────


class TestPersistence:
    def test_new_instance_loads_from_disk(self, tmp_path):
        path = tmp_path / "goals.json"
        sched1 = GoalScheduler(path=path)
        g = sched1.create_goal("persist me", success_criteria=["k1", "k2"])
        # new scheduler instance pointing at same file
        sched2 = GoalScheduler(path=path)
        loaded = sched2.get_goal(g.id)
        assert loaded is not None
        assert loaded.objective == "persist me"
        assert loaded.success_criteria == ["k1", "k2"]

    def test_corrupt_file_does_not_crash(self, tmp_path):
        path = tmp_path / "goals.json"
        path.write_text("{not valid json", encoding="utf-8")
        sched = GoalScheduler(path=path)
        assert sched.list_goals() == []

    def test_empty_criteria_list_persists(self, tmp_path):
        path = tmp_path / "goals.json"
        sched1 = GoalScheduler(path=path)
        g = sched1.create_goal("no criteria", success_criteria=[])
        sched2 = GoalScheduler(path=path)
        loaded = sched2.get_goal(g.id)
        assert loaded.success_criteria == []


# ── check_completion ─────────────────────────────────────────────────────────


class TestCheckCompletion:
    def test_single_criterion_present(self):
        g = Goal(id="g", objective="o", success_criteria=["tests_passed"])
        validation = {"tests_passed": True, "r_phys": 0.85}
        assert GoalScheduler.check_completion(g, validation) is True

    def test_single_criterion_absent(self):
        g = Goal(id="g", objective="o", success_criteria=["converged"])
        validation = {"tests_passed": True}
        assert GoalScheduler.check_completion(g, validation) is False

    def test_multiple_criteria_all_present(self):
        g = Goal(id="g", objective="o", success_criteria=["tests_passed", "r_phys"])
        validation = {"tests_passed": True, "r_phys": 0.9}
        assert GoalScheduler.check_completion(g, validation) is True

    def test_multiple_criteria_one_missing(self):
        g = Goal(id="g", objective="o", success_criteria=["tests_passed", "converged"])
        validation = {"tests_passed": True, "r_phys": 0.9}
        assert GoalScheduler.check_completion(g, validation) is False

    def test_case_insensitive_matching(self):
        g = Goal(id="g", objective="o", success_criteria=["TESTS_PASSED"])
        validation = {"tests_passed": True}
        assert GoalScheduler.check_completion(g, validation) is True

    def test_empty_criteria_returns_false(self):
        g = Goal(id="g", objective="o", success_criteria=[])
        assert GoalScheduler.check_completion(g, {"tests_passed": True}) is False

    def test_none_validation_returns_false(self):
        g = Goal(id="g", objective="o", success_criteria=["x"])
        assert GoalScheduler.check_completion(g, None) is False

    def test_string_validation(self):
        g = Goal(id="g", objective="o", success_criteria=["converged"])
        assert GoalScheduler.check_completion(g, "SCF converged in 10 steps") is True

    def test_criterion_as_substring_of_value(self):
        """criterion 'r_phys' matches validation containing 'r_phys': 0.85"""
        g = Goal(id="g", objective="o", success_criteria=["r_phys"])
        validation = {"r_phys": 0.85, "mode": "workflow"}
        assert GoalScheduler.check_completion(g, validation) is True


# ── Engine integration ───────────────────────────────────────────────────────


class _DummyTracker:
    def start_task(self, *a, **kw) -> None: ...
    def update(self, *a, **kw) -> None: ...
    def complete(self, *a, **kw) -> None: ...
    def fail(self, *a, **kw) -> None: ...


def _make_engine(tmp_path: Path, monkeypatch) -> AutoloopEngine:
    """Engine with heavy components stubbed (same pattern as budget tests)."""
    monkeypatch.setattr("huginn.autoloop.engine.get_model", lambda s: MagicMock())
    monkeypatch.setattr("huginn.autoloop.engine.MemoryManager", lambda: MagicMock())
    monkeypatch.setattr(
        "huginn.autoloop.engine.ProjectKnowledgeGraph", lambda: MagicMock()
    )
    monkeypatch.setattr("huginn.autoloop.engine.BenchmarkRunner", lambda: MagicMock())
    monkeypatch.setattr("huginn.autoloop.engine.CoderRunner", lambda: MagicMock())
    monkeypatch.setattr(
        "huginn.agents.speculator.on_turn_start",
        lambda *a, **kw: {"hint": "", "predictions": []},
        raising=False,
    )
    eng = AutoloopEngine(workspace=tmp_path)
    eng.progress_tracker = _DummyTracker()
    return eng


def _patch_phases(engine: AutoloopEngine, plan_mode: str = "coder") -> None:
    """Replace every phase method with a canned return."""
    engine._perceive = lambda: {"changed_files": ["x.py"], "timestamp": "t"}  # type: ignore[assignment]
    engine._hypothesize = AsyncMock(return_value="h")  # type: ignore[assignment]
    engine._plan = AsyncMock(return_value={"mode": plan_mode, "description": "d"})  # type: ignore[assignment]
    engine._execute = AsyncMock(return_value={"mode": plan_mode, "status": "ok"})  # type: ignore[assignment]
    engine._validate = AsyncMock(return_value={"tests_passed": True})  # type: ignore[assignment]
    engine._learn = AsyncMock(return_value=None)  # type: ignore[assignment]
    engine._report = AsyncMock(return_value=str(engine.workspace / "r.md"))  # type: ignore[assignment]


@pytest.fixture
def no_sleep(monkeypatch):
    async def _noop(*a, **kw):
        return None
    monkeypatch.setattr(asyncio, "sleep", _noop)


class TestEngineGoalIntegration:
    def test_goal_met_stops_early(self, tmp_path, monkeypatch, no_sleep):
        engine = _make_engine(tmp_path, monkeypatch)
        _patch_phases(engine)
        # validation always contains "tests_passed" → criterion met on iter 1
        goal = Goal(
            id="g_early",
            objective="stop fast",
            success_criteria=["tests_passed"],
        )
        result = asyncio.run(engine.run(
            objective="stop fast",
            max_iterations=5,
            goal=goal,
        ))
        # only 1 iteration ran (perceive→hypothesize→plan→execute→validate→learn→stop)
        assert engine._iteration == 1
        assert goal.status == "completed"

    def test_no_goal_runs_all_iterations(self, tmp_path, monkeypatch, no_sleep):
        engine = _make_engine(tmp_path, monkeypatch)
        _patch_phases(engine)
        result = asyncio.run(engine.run(
            objective="run full",
            max_iterations=3,
        ))
        assert engine._iteration == 3

    def test_goal_not_met_runs_all_iterations(self, tmp_path, monkeypatch, no_sleep):
        engine = _make_engine(tmp_path, monkeypatch)
        _patch_phases(engine)
        # criterion "converged" never appears in {"tests_passed": True}
        goal = Goal(
            id="g_never",
            objective="never satisfied",
            success_criteria=["converged"],
        )
        result = asyncio.run(engine.run(
            objective="never satisfied",
            max_iterations=3,
            goal=goal,
        ))
        assert engine._iteration == 3
        assert goal.status != "completed"

    def test_goal_activates_on_run_start(self, tmp_path, monkeypatch, no_sleep):
        engine = _make_engine(tmp_path, monkeypatch)
        _patch_phases(engine)
        goal = Goal(
            id="g_pending",
            objective="activate me",
            success_criteria=["tests_passed"],
            status="pending",
        )
        asyncio.run(engine.run(
            objective="activate me",
            max_iterations=2,
            goal=goal,
        ))
        # status flips to active then completed
        assert goal.status == "completed"

    def test_goal_with_scheduler_persists_completion(self, tmp_path, monkeypatch, no_sleep):
        path = tmp_path / "goals.json"
        scheduler = GoalScheduler(path=path)
        goal = scheduler.create_goal(
            "persisted goal", success_criteria=["tests_passed"]
        )
        engine = _make_engine(tmp_path, monkeypatch)
        engine._goal_scheduler = scheduler
        _patch_phases(engine)
        asyncio.run(engine.run(
            objective="persisted goal",
            max_iterations=3,
            goal=goal,
        ))
        # reload from disk — completion must survive
        scheduler2 = GoalScheduler(path=path)
        loaded = scheduler2.get_goal(goal.id)
        assert loaded.status == "completed"
        assert loaded.completed_at is not None
