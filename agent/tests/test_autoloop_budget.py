"""Tests for the progressive budget (W2 R1).

Locks the behaviour added in R1:
- ProgressiveBudget tier boundaries (open 1-10, medium 11-30, light 31-50)
- IterationBudget.allows truth table
- AutoloopEngine._check_budget unit behaviour (pass / reject / degrade / clear)
- run() integration: workflow rejected in medium+light tiers, coder always ok
- progressive_budget=False disables tiering entirely
- max_iterations default bumped 20 -> 50, progressive_budget default True

All LLM / network / subprocess paths are stubbed; asyncio.sleep is neutralised
so fast-forwarding through skipped iterations is instant.
"""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from huginn.autoloop.budget import IterationBudget, ProgressiveBudget
from huginn.autoloop.engine import AutoloopEngine


# ── shared fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AutoloopEngine:
    """Engine with every heavy sub-component stubbed (same shape as
    test_autoloop_engine.py). run() only touches phase methods, which each
    test overrides per-case."""
    monkeypatch.setattr("huginn.autoloop.engine.get_model", lambda s: MagicMock())
    monkeypatch.setattr("huginn.autoloop.engine.MemoryManager", lambda *a, **kw: MagicMock())
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
    eng = AutoloopEngine(workspace=tmp_path)
    eng.progress_tracker = _DummyTracker()
    return eng


class _DummyTracker:
    def start_task(self, *a, **kw) -> None: ...
    def update(self, *a, **kw) -> None: ...
    def complete(self, *a, **kw) -> None: ...
    def fail(self, *a, **kw) -> None: ...


def _patch_phases(engine: AutoloopEngine, plan_mode: str = "coder") -> None:
    """Replace every phase method with a canned return."""
    engine._perceive = lambda: {"changed_files": ["x.py"], "timestamp": "t"}  # type: ignore[assignment]
    engine._hypothesize = AsyncMock(return_value="h")  # type: ignore[assignment]
    engine._plan = AsyncMock(return_value={"mode": plan_mode, "description": "d"})  # type: ignore[assignment]
    engine._execute = AsyncMock(return_value={"mode": plan_mode, "status": "ok"})  # type: ignore[assignment]
    engine._validate = AsyncMock(return_value={"tests_passed": True})  # type: ignore[assignment]
    engine._learn = AsyncMock(return_value=None)  # type: ignore[assignment]
    engine._report = AsyncMock(return_value=str(engine.workspace / "r.md"))  # type: ignore[assignment]
    # ponytail: 这三个 inter-phase 编排 helper 都会阻塞 event loop:
    #   _blind_spot_pass -> _llm_chat -> await MagicMock.ainvoke (TypeError 被吞)
    #   _maybe_clarify("plan", workflow) -> mgr.ask(timeout=60) 等用户输入
    #   _wait_if_checkpoint_pending -> 600s 轮询 pending_human_review
    # budget 测的是 tier 逻辑, 不关心这些, 全部短路.
    engine._blind_spot_pass = AsyncMock(return_value=[])  # type: ignore[assignment]
    engine._maybe_clarify = AsyncMock(return_value=None)  # type: ignore[assignment]
    engine._wait_if_checkpoint_pending = AsyncMock(return_value=None)  # type: ignore[assignment]


def _fast_forward_perceive(skip_until: int):
    """Return a perceive fn that yields None for the first skip_until-1 calls
    then a real context. Lets a test jump straight to iteration `skip_until`."""
    counter = {"n": 0}

    def _perceive():
        counter["n"] += 1
        if counter["n"] < skip_until:
            return None
        return {"changed_files": ["x.py"], "timestamp": "t"}

    return _perceive


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralise asyncio.sleep so skipped iterations don't cost 2s each."""
    async def _noop(*a, **kw):  # noqa: ANN202
        return None

    monkeypatch.setattr(asyncio, "sleep", _noop)


# ── budget dataclass unit tests ──────────────────────────────────────────────


class TestProgressiveBudgetTiers:
    def test_open_tier_1_to_10(self):
        b = ProgressiveBudget.default()
        for n in (1, 5, 10):
            assert b.for_iteration(n).label == "open"

    def test_medium_tier_11_to_30(self):
        b = ProgressiveBudget.default()
        for n in (11, 20, 30):
            assert b.for_iteration(n).label == "medium"

    def test_light_tier_31_to_50(self):
        b = ProgressiveBudget.default()
        for n in (31, 40, 50):
            assert b.for_iteration(n).label == "light"

    def test_past_last_bound_falls_back_to_open(self):
        # runaway loop past the last tier bound should not crash — fall back
        # to open so the agent can still make progress.
        b = ProgressiveBudget.default()
        assert b.for_iteration(99).label == "open"

    def test_medium_tier_modes(self):
        b = ProgressiveBudget.default()
        tier = b.for_iteration(20)
        assert tier.allowed_modes == ("coder", "explore")
        assert tier.max_calls == 30

    def test_light_tier_modes(self):
        b = ProgressiveBudget.default()
        tier = b.for_iteration(40)
        assert tier.allowed_modes == ("coder",)
        assert tier.max_calls == 20

    def test_open_tier_no_restriction(self):
        b = ProgressiveBudget.default()
        tier = b.for_iteration(5)
        assert tier.allowed_modes is None
        assert tier.max_calls is None


class TestIterationBudgetAllows:
    def test_none_allowed_modes_allows_everything(self):
        b = IterationBudget(allowed_modes=None, max_calls=None, label="x")
        assert b.allows("workflow")
        assert b.allows("coder")
        assert b.allows(None)
        assert b.allows("anything_weird")

    def test_restricted_modes(self):
        b = IterationBudget(allowed_modes=("coder",), max_calls=5, label="light")
        assert b.allows("coder")
        assert not b.allows("workflow")
        assert not b.allows("explore")

    def test_none_mode_rejected_when_restricted(self):
        b = IterationBudget(allowed_modes=("coder",), max_calls=5, label="light")
        assert not b.allows(None)


# ── _check_budget unit tests ─────────────────────────────────────────────────


class TestCheckBudgetUnit:
    def _fresh(self, engine: AutoloopEngine) -> None:
        engine._budget = ProgressiveBudget.default()
        engine._budget_degraded = False
        engine._budget_rejects = {}
        engine._speculator_hint = ""

    def test_budget_none_always_passes(self, engine: AutoloopEngine):
        engine._budget = None
        engine._budget_degraded = False
        assert engine._check_budget(16, {"mode": "workflow"}) is True

    def test_degraded_always_passes(self, engine: AutoloopEngine):
        self._fresh(engine)
        engine._budget_degraded = True
        assert engine._check_budget(16, {"mode": "workflow"}) is True

    def test_open_tier_allows_workflow(self, engine: AutoloopEngine):
        self._fresh(engine)
        assert engine._check_budget(5, {"mode": "workflow"}) is True
        assert engine._speculator_hint == ""

    def test_medium_tier_rejects_workflow(self, engine: AutoloopEngine):
        self._fresh(engine)
        assert engine._check_budget(20, {"mode": "workflow"}) is False
        assert "medium" in engine._speculator_hint
        assert "workflow" in engine._speculator_hint
        assert engine._budget_rejects.get("medium") == 1

    def test_medium_tier_allows_coder_and_explore(self, engine: AutoloopEngine):
        self._fresh(engine)
        assert engine._check_budget(20, {"mode": "coder"}) is True
        assert engine._check_budget(20, {"mode": "explore"}) is True

    def test_light_tier_rejects_explore(self, engine: AutoloopEngine):
        self._fresh(engine)
        assert engine._check_budget(40, {"mode": "explore"}) is False
        assert engine._budget_rejects.get("light") == 1

    def test_light_tier_allows_coder(self, engine: AutoloopEngine):
        self._fresh(engine)
        assert engine._check_budget(40, {"mode": "coder"}) is True

    def test_pass_clears_reject_counter(self, engine: AutoloopEngine):
        self._fresh(engine)
        engine._budget_rejects = {"medium": 3}
        # a passing call wipes the counter for that tier
        assert engine._check_budget(20, {"mode": "coder"}) is True
        assert "medium" not in engine._budget_rejects

    def test_degrade_after_max_calls(self, engine: AutoloopEngine):
        self._fresh(engine)
        # light tier max_calls=20: 20 rejects still blocked, 21st degrades.
        for _ in range(20):
            assert engine._check_budget(40, {"mode": "workflow"}) is False
        assert engine._budget_degraded is False
        # 21st reject hits the cap -> degrade + allow
        assert engine._check_budget(40, {"mode": "workflow"}) is True
        assert engine._budget_degraded is True
        # subsequent calls pass because degraded flag sticks
        assert engine._check_budget(40, {"mode": "workflow"}) is True

    def test_medium_tier_degrade_cap_is_30(self, engine: AutoloopEngine):
        self._fresh(engine)
        for _ in range(30):
            assert engine._check_budget(20, {"mode": "workflow"}) is False
        assert engine._budget_degraded is False
        assert engine._check_budget(20, {"mode": "workflow"}) is True
        assert engine._budget_degraded is True


# ── run() integration ────────────────────────────────────────────────────────


class TestEngineRunBudgetIntegration:
    def test_workflow_allowed_in_open_tier(
        self, engine: AutoloopEngine, no_sleep
    ):
        _patch_phases(engine, plan_mode="workflow")
        result = asyncio.run(engine.run(objective="o", max_iterations=1))
        engine._execute.assert_called_once()
        assert result.success is True

    def test_workflow_rejected_in_medium_tier(
        self, engine: AutoloopEngine, no_sleep
    ):
        # _patch_phases first (it resets _perceive), then override with the
        # fast-forward fn so iters 1-10 skip and iter 11 hits the medium tier.
        _patch_phases(engine, plan_mode="workflow")
        engine._perceive = _fast_forward_perceive(skip_until=11)  # type: ignore[assignment]
        asyncio.run(engine.run(objective="o", max_iterations=11))
        # budget rejected workflow at iter 11 -> execute never called
        engine._execute.assert_not_called()
        # hint carries the budget feedback for the next iteration's prompt
        assert "medium" in engine._speculator_hint
        assert "workflow" in engine._speculator_hint

    def test_workflow_rejected_in_light_tier(
        self, engine: AutoloopEngine, no_sleep
    ):
        _patch_phases(engine, plan_mode="workflow")
        engine._perceive = _fast_forward_perceive(skip_until=31)  # type: ignore[assignment]
        asyncio.run(engine.run(objective="o", max_iterations=31))
        engine._execute.assert_not_called()
        engine._learn.assert_not_called()
        engine._validate.assert_not_called()
        assert "light" in engine._speculator_hint

    def test_progressive_budget_disabled_allows_workflow_at_iter_11(
        self, engine: AutoloopEngine, no_sleep
    ):
        _patch_phases(engine, plan_mode="workflow")
        engine._perceive = _fast_forward_perceive(skip_until=11)  # type: ignore[assignment]
        asyncio.run(engine.run(
            objective="o", max_iterations=11, progressive_budget=False,
        ))
        # tiering off -> workflow reaches execute at iter 11
        engine._execute.assert_called_once()
        assert engine._budget is None

    def test_coder_allowed_in_light_tier(
        self, engine: AutoloopEngine, no_sleep
    ):
        # coder is the only mode allowed in light tier -> reaches execute
        _patch_phases(engine, plan_mode="coder")
        engine._perceive = _fast_forward_perceive(skip_until=31)  # type: ignore[assignment]
        asyncio.run(engine.run(objective="o", max_iterations=31))
        engine._execute.assert_called_once()

    def test_explore_allowed_in_medium_but_not_light(
        self, engine: AutoloopEngine, no_sleep
    ):
        # medium tier (iter 11): explore allowed -> execute called
        _patch_phases(engine, plan_mode="explore")
        engine._perceive = _fast_forward_perceive(skip_until=11)  # type: ignore[assignment]
        asyncio.run(engine.run(objective="o", max_iterations=11))
        engine._execute.assert_called_once()

    def test_explore_rejected_in_light_tier(
        self, engine: AutoloopEngine, no_sleep
    ):
        _patch_phases(engine, plan_mode="explore")
        engine._perceive = _fast_forward_perceive(skip_until=31)  # type: ignore[assignment]
        asyncio.run(engine.run(objective="o", max_iterations=31))
        engine._execute.assert_not_called()

    def test_budget_and_gate_compose(
        self, engine: AutoloopEngine, no_sleep
    ):
        """Budget runs before the phase-gate. A plan that passes the budget
        still has to clear the gate (mode+description evidence). Verify both
        checks fire independently in one run."""
        _patch_phases(engine, plan_mode="coder")
        engine._perceive = _fast_forward_perceive(skip_until=11)  # type: ignore[assignment]
        # coder passes medium budget, but empty description fails the gate
        engine._plan = AsyncMock(return_value={"mode": "coder", "description": ""})  # type: ignore[assignment]
        asyncio.run(engine.run(objective="o", max_iterations=11))
        # budget passed (coder in medium) but gate blocked (empty description)
        # -> execute never called
        engine._execute.assert_not_called()
        # gate feedback is in the hint
        assert "缺" in engine._speculator_hint or "description" in engine._speculator_hint.lower()

    def test_default_max_iterations_is_50(self):
        sig = inspect.signature(AutoloopEngine.run)
        assert sig.parameters["max_iterations"].default == 50

    def test_default_progressive_budget_is_true(self):
        sig = inspect.signature(AutoloopEngine.run)
        assert sig.parameters["progressive_budget"].default is True


# ── phase-gate doesn't burn budget rejection quota ───────────────────────────


class TestGateDoesNotBurnBudgetQuota:
    """A phase-gate block is a separate path from a budget reject. Verify the
    budget reject counter only advances on actual budget rejects, not on gate
    blocks — so a gate-blocked iteration in the light tier still has its full
    max_calls quota available for real budget rejects."""

    def test_gate_block_keeps_reject_counter_at_zero(
        self, engine: AutoloopEngine, no_sleep
    ):
        _patch_phases(engine, plan_mode="coder")
        engine._perceive = _fast_forward_perceive(skip_until=31)  # type: ignore[assignment]
        # coder passes budget, but empty description fails the gate
        engine._plan = AsyncMock(return_value={"mode": "coder", "description": ""})  # type: ignore[assignment]
        asyncio.run(engine.run(objective="o", max_iterations=31))
        # gate blocked (empty description) but budget passed (coder) -> no reject
        assert engine._budget_rejects.get("light", 0) == 0
        assert engine._budget_degraded is False
        engine._execute.assert_not_called()
