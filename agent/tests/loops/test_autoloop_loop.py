"""P0 integration tests for the autoloop engine (autoloop/engine.py).

Drives the real AutoloopEngine.run() end-to-end with a FakeLLM in
callable mode. Heavy external calls (BenchmarkRunner, CoderRunner,
PerceptionLayer) are stubbed — the LLM decision path is never mocked.

Test matrix:
  1. Full 7-phase pipeline completes
  2. Phase gate blocks invalid plan → retry succeeds
  3. Iteration budget (max_iterations) limits the loop
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from huginn.autoloop.engine import AutoloopEngine, AutoloopResult
from huginn.autoloop.phase_gate import get_shared_phase_gate_state
from huginn.memory.manager import MemoryManager
from tests.fixtures.fake_llm import make_callable_llm

# ── shared helpers (trimmed from test_autoloop_e2e) ───────────────


class _DummyTracker:
    def start_task(self, *a, **kw): ...
    def update(self, *a, **kw): ...
    def complete(self, *a, **kw): ...
    def fail(self, *a, **kw): ...


class _StubBenchReport:
    passed = 2
    failed = 0
    skipped = 0


class _StubBenchRunner:
    def run(self, categories=None):
        return _StubBenchReport()


def _make_stage_llm(plan_fn=None):
    """Build a callable FakeLLM that routes by prompt content.

    *plan_fn* lets the caller override the plan response (for gate tests).
    """
    def respond(prompt: str) -> str:
        low = prompt.lower()

        if "testable hypothesis" in low:
            return (
                "If the Ca/Si ratio in C-S-H increases, then the "
                "interlayer spacing decreases, accelerating water "
                "diffusion through the gel pores."
            )

        if "choose one mode" in low:
            if plan_fn is not None:
                return plan_fn()
            return (
                "MODE: coder\n"
                "DESCRIPTION: Parameterize the Ca/Si ratio in the "
                "diffusion analysis script and add a convergence check."
            )

        if "critical peer reviewer" in low or "point out" in low:
            return (
                "The coder output lacks a convergence check on system "
                "size. Next step: add a convergence study."
            )

        if "graduate student" in low or "pedagogical" in low:
            return (
                "This loop tested whether the Ca/Si ratio affects C-S-H "
                "water diffusion. The coder stage updated the analysis "
                "script. Validation confirmed the approach."
            )

        if "修正假设" in prompt or "refined hypothesis" in low:
            return "Refined: percolation-threshold model applies at high Ca/Si."

        return "[]"

    return make_callable_llm(respond, name="loop-test-llm")


def _stub_heavy_calls(monkeypatch, fake_llm):
    monkeypatch.setattr("huginn.autoloop.engine.get_model", lambda settings: fake_llm)
    monkeypatch.setattr("huginn.autoloop.engine.BenchmarkRunner", lambda: _StubBenchRunner())
    monkeypatch.setattr("huginn.autoloop.engine.CoderRunner", lambda *a, **kw: MagicMock())
    monkeypatch.setattr("huginn.autoloop.engine.ProjectKnowledgeGraph", lambda *a, **kw: MagicMock())
    monkeypatch.setattr(
        "huginn.agents.speculator.on_turn_start",
        lambda *a, **kw: {"hint": "", "predictions": []},
        raising=False,
    )
    # ponytail: KB 第一次懒加载会 seed_knowledge_base 跑 ONNX 嵌入,
    # CI 首次冷启动 > 120s timeout. _build_kb_text 已处理 None, 直接跳过.
    monkeypatch.setattr("huginn.autoloop.engine.AutoloopEngine._get_kb", lambda self: None)


def _bypass_validate_gate():
    state = get_shared_phase_gate_state()
    state.overrides.add(("validate", "learn"))
    return state


def _restore_gate(state):
    state.overrides.discard(("validate", "learn"))
    state.history.clear()
    state.pending_transition = None


def _make_engine(tmp_path, fake_llm, monkeypatch):
    _stub_heavy_calls(monkeypatch, fake_llm)
    memory = MemoryManager()
    engine = AutoloopEngine(
        workspace=tmp_path,
        verification_model=fake_llm,
        memory_manager=memory,
    )
    engine.progress_tracker = _DummyTracker()
    engine._perceive = lambda: {
        "changed_files": ["diffusion_analysis.py"],
        "git_diff": "+def calc_diffusion(ca_si_ratio): ...",
        "timestamp": "2026-07-04T10:00:00Z",
        "goal": "Optimize C-S-H defect kinetics",
    }
    return engine, memory


# ── 1. full 7-phase: perceive → hypothesize → plan → execute → ... ─


class TestAutoloopFullCycle:
    @pytest.mark.asyncio
    async def test_all_seven_phases_complete(self, tmp_path, monkeypatch):
        """Full pipeline: 6 stages + report = 7 phases, all completed."""
        fake_llm = _make_stage_llm()
        engine, memory = _make_engine(tmp_path, fake_llm, monkeypatch)

        (tmp_path / "test_smoke.py").write_text(
            "def test_ok():\n    assert True\n", encoding="utf-8"
        )
        gate_state = _bypass_validate_gate()
        try:
            result = await engine.run(
                objective="Optimize C-S-H defect kinetics",
                max_iterations=1,
                progressive_budget=False,
            )
        finally:
            _restore_gate(gate_state)

        assert isinstance(result, AutoloopResult)
        assert len(result.phases) == 7
        names = [p.name for p in result.phases]
        assert names == [
            "perceive", "hypothesize", "plan", "execute",
            "validate", "learn", "report",
        ]
        for phase in result.phases:
            assert phase.status == "completed", (
                f"phase '{phase.name}' status={phase.status}"
            )
        # LLM was actually called (not mocked)
        assert fake_llm.call_count >= 3


# ── 2. phase gate: invalid plan → rejected → retry ────────────────


class TestAutoloopPhaseGate:
    @pytest.mark.asyncio
    async def test_gate_blocks_then_retries(self, tmp_path, monkeypatch):
        """First plan has empty description → gate blocks → second plan passes."""
        plan_calls = [0]

        def plan_fn():
            plan_calls[0] += 1
            if plan_calls[0] == 1:
                # empty string → _parse_plan gives description="" → gate blocks
                return ""
            return (
                "MODE: coder\n"
                "DESCRIPTION: Add a convergence check to the script."
            )

        fake_llm = _make_stage_llm(plan_fn=plan_fn)
        engine, _ = _make_engine(tmp_path, fake_llm, monkeypatch)

        (tmp_path / "test_smoke.py").write_text(
            "def test_ok():\n    assert True\n", encoding="utf-8"
        )
        gate_state = _bypass_validate_gate()
        try:
            result = await engine.run(
                objective="Optimize C-S-H defect kinetics",
                max_iterations=3,
                progressive_budget=False,
            )
        finally:
            _restore_gate(gate_state)

        # First plan was blocked (empty description), second succeeded.
        # So we should see more than 7 phases (at least 2 plan attempts).
        plan_phases = [p for p in result.phases if p.name == "plan"]
        assert len(plan_phases) >= 2, (
            f"expected >= 2 plan phases (gate retry), got {len(plan_phases)}"
        )

        # First plan had empty description (gate blocked it)
        first_plan = plan_phases[0].result
        assert first_plan["description"] == ""

        # A later plan had a real description and the pipeline continued
        last_plan = plan_phases[-1].result
        assert last_plan["description"] != ""

        # Execute phase exists (pipeline got past the gate on retry)
        execute_phases = [p for p in result.phases if p.name == "execute"]
        assert len(execute_phases) >= 1

        # Report still ran
        report_phases = [p for p in result.phases if p.name == "report"]
        assert len(report_phases) >= 1


# ── 3. budget exhaustion: max_iterations limit → stops ─────────────


class TestAutoloopBudgetExhaustion:
    @pytest.mark.asyncio
    async def test_max_iterations_limits_loop(self, tmp_path, monkeypatch):
        """max_iterations=1 → exactly one iteration, then stops."""
        fake_llm = _make_stage_llm()
        engine, _ = _make_engine(tmp_path, fake_llm, monkeypatch)

        (tmp_path / "test_smoke.py").write_text(
            "def test_ok():\n    assert True\n", encoding="utf-8"
        )
        gate_state = _bypass_validate_gate()
        try:
            result = await engine.run(
                objective="Optimize C-S-H defect kinetics",
                max_iterations=1,
                progressive_budget=False,
            )
        finally:
            _restore_gate(gate_state)

        # One iteration = 6 stages + 1 report = 7 phases
        assert len(result.phases) == 7

        # Exactly one perceive phase (one iteration)
        perceive_count = sum(1 for p in result.phases if p.name == "perceive")
        assert perceive_count == 1

    @pytest.mark.asyncio
    async def test_progressive_budget_rejects_expensive_mode(self, tmp_path, monkeypatch):
        """Workflow mode is rejected at medium tier (iteration 6+).

        ponytail: We don't run 6 full iterations (too slow). Instead we
        verify the budget mechanism directly: feed a 'workflow' plan at
        iteration 6 and check _check_budget returns False.
        """
        from huginn.autoloop.budget import ProgressiveBudget

        budget = ProgressiveBudget.default()
        engine, _ = _make_engine(tmp_path, _make_stage_llm(), monkeypatch)
        engine._budget = budget
        engine._budget_degraded = False
        engine._budget_rejects = {}

        # Iteration 6 falls in the 'medium' tier where 'workflow' is not allowed
        allowed = engine._check_budget(6, {"mode": "workflow"})
        assert allowed is False, "workflow should be rejected at iteration 6"

        # 'coder' is always allowed
        allowed_coder = engine._check_budget(6, {"mode": "coder"})
        assert allowed_coder is True

        # At iteration 1 (open tier), 'workflow' is allowed
        allowed_early = engine._check_budget(1, {"mode": "workflow"})
        assert allowed_early is True
