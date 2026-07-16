"""Tests for AutoloopEngine <-> PlanStore integration.

Locks the wiring added when _plan() started persisting plans to PlanStore
and running the confirm/reject gate, plus the complete_plan call after
execute and the store_plan_progress call in _learn.

All LLM / network / subprocess paths are stubbed so the tests are hermetic.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from huginn.autoloop.engine import AutoloopEngine
from huginn.autoloop.plan_store import PlanStore


class _DummyTracker:
    """Minimal ProgressTracker stand-in -- just absorbs calls."""

    def start_task(self, *a, **kw) -> None: ...
    def update(self, *a, **kw) -> None: ...
    def complete(self, *a, **kw) -> None: ...
    def fail(self, *a, **kw) -> None: ...


@pytest.fixture
def engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AutoloopEngine:
    """Engine with heavy sub-components stubbed and a real PlanStore on tmp_path."""
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
    monkeypatch.setattr(
        "huginn.agents.speculator.on_turn_start",
        lambda *a, **kw: {"hint": "", "predictions": []},
        raising=False,
    )
    # ponytail: KB 冷启动跑 ONNX embedding > 120s, KG 写 ~/.huginn 污染 home
    monkeypatch.setattr("huginn.autoloop.engine.AutoloopEngine._get_kb", lambda self: None)
    monkeypatch.setattr("huginn.autoloop.conjecture.get_kg", lambda *a, **kw: None)
    eng = AutoloopEngine(workspace=tmp_path)
    eng.progress_tracker = _DummyTracker()
    # real PlanStore on a temp file so we can inspect persisted state
    eng._plan_store = PlanStore(path=tmp_path / "plans.json")
    # _build_plan_prompt pulls KB/KG/evolution -- not under test, short-circuit it
    eng._build_plan_prompt = lambda hyp, ctx: "prompt"  # type: ignore[assignment]
    return eng


# ── _plan persists to PlanStore ──────────────────────────────────


class TestPlanCreatesPlanstoreEntry:
    def test_plan_creates_planstore_entry(self, engine: AutoloopEngine):
        engine._llm_chat = AsyncMock(  # type: ignore[assignment]
            return_value="MODE: coder\nDESCRIPTION: tweak x"
        )
        # non-expensive mode -> _maybe_clarify returns None -> auto-confirm
        engine._maybe_clarify = AsyncMock(return_value=None)  # type: ignore[assignment]

        plan = asyncio.run(engine._plan("test hypothesis", {}))

        assert plan is not None
        assert "plan_id" in plan
        plans = engine._plan_store.list_plans()
        assert len(plans) == 1
        assert plans[0].objective == "test hypothesis"
        # confirmed + mark_executing -> status is "executing"
        assert plans[0].status == "executing"
        assert plans[0].confirmed_at is not None


class TestPlanConfirmedWhenUserAgrees:
    def test_plan_confirmed_when_user_agrees(self, engine: AutoloopEngine):
        engine._llm_chat = AsyncMock(  # type: ignore[assignment]
            return_value="MODE: workflow\nDESCRIPTION: run DFT relax"
        )
        # user explicitly says yes
        engine._maybe_clarify = AsyncMock(return_value=True)  # type: ignore[assignment]

        plan = asyncio.run(engine._plan("relax GaN", {}))

        assert plan is not None
        persisted = engine._plan_store.get_plan(plan["plan_id"])
        assert persisted is not None
        assert persisted.status == "executing"
        assert persisted.confirmed_at is not None


class TestPlanRejectedWhenUserDeclines:
    def test_plan_rejected_when_user_declines(self, engine: AutoloopEngine):
        engine._llm_chat = AsyncMock(  # type: ignore[assignment]
            return_value="MODE: workflow\nDESCRIPTION: run DFT relax"
        )
        # user says no
        engine._maybe_clarify = AsyncMock(return_value=False)  # type: ignore[assignment]

        plan = asyncio.run(engine._plan("relax GaN", {}))

        assert plan is None  # rejected -> _plan returns None
        plans = engine._plan_store.list_plans()
        assert len(plans) == 1
        assert plans[0].status == "abandoned"
        assert plans[0].reject_reason == "user declined"


# ── run() marks plan complete after execute ──────────────────────


class TestPlanCompletedAfterExecute:
    def test_plan_completed_after_execute(self, engine: AutoloopEngine):
        # real _plan (creates PlanStore entry), everything else mocked
        engine._llm_chat = AsyncMock(  # type: ignore[assignment]
            return_value="MODE: coder\nDESCRIPTION: do x"
        )
        engine._maybe_clarify = AsyncMock(return_value=None)  # type: ignore[assignment]
        engine._perceive = lambda: {"changed_files": ["x.py"], "timestamp": "t"}  # type: ignore[assignment]
        engine._hypothesize = AsyncMock(return_value="test hypothesis")  # type: ignore[assignment]
        engine._execute = AsyncMock(  # type: ignore[assignment]
            return_value={"mode": "coder", "status": "ok"}
        )
        engine._validate = AsyncMock(return_value={"tests_passed": True})  # type: ignore[assignment]
        engine._learn = AsyncMock(return_value=None)  # type: ignore[assignment]
        engine._report = AsyncMock(  # type: ignore[assignment]
            return_value=str(engine.workspace / "r.md")
        )

        result = asyncio.run(engine.run(objective="o", max_iterations=1))

        assert result.success is True
        plans = engine._plan_store.list_plans()
        assert len(plans) == 1
        assert plans[0].status == "completed"
        assert plans[0].completed_at is not None


# ── _learn stores plan progress to memory ────────────────────────


class TestPlanProgressStoredInMemory:
    def test_plan_progress_stored_in_memory(self, engine: AutoloopEngine):
        # create a plan via _plan so we get a real plan_id in the store
        engine._llm_chat = AsyncMock(  # type: ignore[assignment]
            return_value="MODE: coder\nDESCRIPTION: do x"
        )
        engine._maybe_clarify = AsyncMock(return_value=None)  # type: ignore[assignment]

        plan = asyncio.run(engine._plan("some hypothesis", {}))
        assert plan is not None
        assert "plan_id" in plan

        # memory is a MagicMock (from the fixture), so we can assert on calls.
        # run the real _learn -- it should hit store_plan_progress.
        asyncio.run(
            engine._learn("some hypothesis", plan, {"tests_passed": True})
        )

        engine.memory.store_plan_progress.assert_called_once()
        kwargs = engine.memory.store_plan_progress.call_args.kwargs
        assert kwargs["plan_id"] == plan["plan_id"]
        assert kwargs["objective"] == "some hypothesis"
        assert kwargs["status"] == "executing"
