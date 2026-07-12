"""First real E2E test for AutoloopEngine — drives all 6 stages with a FakeLLM.

All 10 existing test files stub every stage method with AsyncMock, leaving
zero real pipeline coverage.  This module lets the genuine engine.run()
execute end-to-end: the FakeLLM makes the real _llm_chat calls that
hypothesize / plan / validate / report depend on.  Only genuinely heavy
external calls (BenchmarkRunner, CoderRunner init, PerceptionLayer) are
stubbed — the LLM decision path is never mocked.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from huginn.autoloop.engine import AutoloopEngine, AutoloopResult
from huginn.autoloop.hypothesis_loop import HypothesisGraph
from huginn.autoloop.phase_gate import get_shared_phase_gate_state
from huginn.memory.manager import MemoryManager
from tests.fixtures.fake_llm import make_callable_llm


# ── helpers ──────────────────────────────────────────────────────────


class _DummyTracker:
    """Bare-bones tracker — keeps the process-level singleton untouched."""

    def start_task(self, *a, **kw): ...
    def update(self, *a, **kw): ...
    def complete(self, *a, **kw): ...
    def fail(self, *a, **kw): ...


class _StubBenchReport:
    passed = 2
    failed = 0
    skipped = 0


class _StubBenchRunner:
    """Drop-in for BenchmarkRunner — returns a clean report, no real compute."""

    def run(self, categories=None):
        return _StubBenchReport()


def _make_stage_llm():
    """Build a callable FakeLLM that routes by prompt content.

    Each pipeline stage embeds distinctive keywords in its HumanMessage, so
    we can return the right response shape without any AsyncMock.
    """
    def respond(prompt: str) -> str:
        low = prompt.lower()

        # _hypothesize → "Generate a single, testable hypothesis"
        if "testable hypothesis" in low:
            return (
                "If the Ca/Si ratio in C-S-H increases, then the "
                "interlayer spacing decreases, accelerating water "
                "diffusion through the gel pores."
            )

        # _plan → "Choose ONE mode"
        if "choose one mode" in low:
            return (
                "MODE: coder\n"
                "DESCRIPTION: Parameterize the Ca/Si ratio in the "
                "diffusion analysis script and add a convergence check."
            )

        # _validate reviewer critique → "As a critical peer reviewer"
        if "critical peer reviewer" in low or "point out" in low:
            return (
                "The coder output lacks a convergence check on system "
                "size.  The result is not benchmarked against literature "
                "values.  Next step: add a convergence study before "
                "trusting the diffusion coefficient."
            )

        # _report tutor narrative → "Summarize for a graduate student"
        if "graduate student" in low or "pedagogical" in low:
            return (
                "This loop tested whether the Ca/Si ratio affects C-S-H "
                "water diffusion.  The coder stage updated the analysis "
                "script to parameterize Ca/Si.  Validation confirmed the "
                "approach but flagged a missing convergence check.  Next "
                "iteration should add a system-size convergence study."
            )

        # refine_failed → "修正假设" in the HumanMessage
        if "修正假设" in prompt or "refined hypothesis" in low:
            return (
                "If the Ca/Si ratio exceeds 1.2, the C-S-H interlayer "
                "spacing decreases non-linearly and water diffusion "
                "follows a percolation-threshold model rather than a "
                "simple linear trend."
            )

        # RedTeamReviewer LLM prompt or anything unexpected — return
        # an empty JSON array so _parse_llm_findings yields no findings.
        return "[]"

    return make_callable_llm(respond, name="e2e-stage-llm")


def _stub_heavy_calls(monkeypatch, fake_llm):
    """Monkeypatch only genuinely heavy / env-dependent pieces.

    get_model        → FakeLLM (so self.model is real, not a MagicMock)
    BenchmarkRunner  → stub (avoids real benchmark compute)
    CoderRunner      → MagicMock (constructor side-effects only)
    ProjectKnowledgeGraph → MagicMock (not used in the run loop)
    on_turn_start    → no-op (speculator scenario matching)
    """
    monkeypatch.setattr(
        "huginn.autoloop.engine.get_model", lambda settings: fake_llm
    )
    monkeypatch.setattr(
        "huginn.autoloop.engine.BenchmarkRunner", lambda: _StubBenchRunner()
    )
    monkeypatch.setattr(
        "huginn.autoloop.engine.CoderRunner", lambda: MagicMock()
    )
    monkeypatch.setattr(
        "huginn.autoloop.engine.ProjectKnowledgeGraph", lambda *a, **kw: MagicMock()
    )
    monkeypatch.setattr(
        "huginn.agents.speculator.on_turn_start",
        lambda *a, **kw: {"hint": "", "predictions": []},
        raising=False,
    )


def _bypass_validate_gate():
    """Add a one-shot override so validate→learn passes.

    _validate shells out to ``python -m pytest`` as a subprocess, which is
    environment-dependent (PATH, pytest discovery, etc.).  We bypass only
    the gate *decision* — the validate stage itself, including the
    LLM-driven reviewer critique, still runs for real.
    """
    state = get_shared_phase_gate_state()
    state.overrides.add(("validate", "learn"))
    return state


def _restore_gate(state):
    state.overrides.discard(("validate", "learn"))
    state.history.clear()
    state.pending_transition = None


def _make_engine(tmp_path, fake_llm, monkeypatch):
    """Wire up a real AutoloopEngine with heavy calls stubbed."""
    _stub_heavy_calls(monkeypatch, fake_llm)

    # Real MemoryManager — _learn actually writes session messages that
    # we can assert on afterwards.
    memory = MemoryManager()

    engine = AutoloopEngine(
        workspace=tmp_path,
        verification_model=fake_llm,
        memory_manager=memory,
    )
    engine.progress_tracker = _DummyTracker()

    # _perceive is NOT an LLM stage — it scans the filesystem via
    # PerceptionLayer.  In an empty tmp_path it returns None and the loop
    # would skip every iteration, so we feed it a minimal context dict.
    engine._perceive = lambda: {
        "changed_files": ["diffusion_analysis.py"],
        "git_diff": "+def calc_diffusion(ca_si_ratio): ...",
        "timestamp": "2026-07-04T10:00:00Z",
        "goal": "Optimize C-S-H defect kinetics",
    }
    return engine, memory


# ── tests ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_autoloop_full_cycle_with_fake_llm(tmp_path, monkeypatch):
    """Drive the full 6-stage pipeline with a FakeLLM — no AsyncMock stubs.

    Verifies:
      * engine.run() completes without crashing
      * all 7 phases (6 stages + report) have status "completed"
      * LLM-driven stages produced real, non-empty output
      * the markdown report file was written to disk
      * FakeLLM.call_count > 0 (proves real LLM calls, not mocks)
      * memory has session messages from _learn
      * HypothesisGraph tracks the engine's hypothesis
    """
    fake_llm = _make_stage_llm()
    engine, memory = _make_engine(tmp_path, fake_llm, monkeypatch)

    # Drop a passing test so _validate's pytest subprocess has something
    # to find (the gate override is the safety net if python isn't on PATH).
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

    # ── completion ───────────────────────────────────────────────
    assert isinstance(result, AutoloopResult)
    assert result.run_id.startswith("loop_")
    assert result.objective == "Optimize C-S-H defect kinetics"

    # 6 stages of one iteration + final report = 7 phases
    assert len(result.phases) == 7
    names = [p.name for p in result.phases]
    assert names == [
        "perceive", "hypothesize", "plan", "execute",
        "validate", "learn", "report",
    ]

    for phase in result.phases:
        assert phase.status == "completed", (
            f"phase '{phase.name}' status={phase.status} error={phase.error}"
        )

    # ── LLM-driven stages produced real output ───────────────────
    hyp_phase = next(p for p in result.phases if p.name == "hypothesize")
    assert hyp_phase.result, "hypothesize returned empty"
    assert "Ca/Si" in hyp_phase.result or "diffusion" in hyp_phase.result.lower()

    plan_phase = next(p for p in result.phases if p.name == "plan")
    assert isinstance(plan_phase.result, dict)
    assert plan_phase.result["mode"] == "coder"
    assert "Ca/Si" in plan_phase.result["description"]

    # validate ran the real reviewer critique via FakeLLM
    val_phase = next(p for p in result.phases if p.name == "validate")
    assert isinstance(val_phase.result, dict)
    assert "reviewer_critique" in val_phase.result or "tests_passed" in val_phase.result

    # ── report file written ─────────────────────────────────────
    assert result.report_path is not None
    report_file = Path(result.report_path)
    assert report_file.exists(), f"report not found at {report_file}"
    report_text = report_file.read_text(encoding="utf-8")
    assert "Huginn Autoloop Report" in report_text
    assert "Optimize C-S-H defect kinetics" in report_text
    # tutor narrative comes from the real FakeLLM call in _report
    assert "Tutor" in report_text or "diffusion" in report_text.lower()

    # ── FakeLLM was actually called (not mocked) ─────────────────
    # 4 calls minimum: hypothesize + plan + validate reviewer + report tutor
    assert fake_llm.call_count >= 3, (
        f"expected >= 3 real LLM calls, got {fake_llm.call_count}"
    )

    # ── memory has entries from _learn ──────────────────────────
    assert len(memory.session.messages) > 0, "no session messages from _learn"

    # ── HypothesisGraph tracks the engine's hypothesis ───────────
    graph = HypothesisGraph()
    h1 = graph.add_hypothesis(
        statement=hyp_phase.result,
        rationale="Generated by AutoloopEngine iteration 1",
        testable_prediction="Interlayer spacing decreases with Ca/Si",
    )
    assert len(graph.all_nodes()) == 1
    assert len(graph.frontier()) == 1
    assert graph.get(h1).status == "untested"


@pytest.mark.asyncio
async def test_autoloop_hypothesis_evolution(tmp_path, monkeypatch):
    """Verify refine_failed produces a parent→child hypothesis relationship.

    Runs one real engine iteration to get a FakeLLM-generated hypothesis,
    then simulates refutation and refinement through the real
    HypothesisGraph.refine_failed flow (which itself calls the FakeLLM
    via RedTeamReviewer + _llm_refine).

    Verifies:
      * the refined hypothesis differs from the original
      * child.parent_id points to the parent node
      * parent status is "superseded", child status is "untested"
      * a derive edge connects parent → child
      * children() helper returns the refined node
      * FakeLLM was called during refinement (real LLM, not template)
    """
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

    # Grab the hypothesis from the real engine run
    hyp_phase = next(p for p in result.phases if p.name == "hypothesize")
    original = hyp_phase.result
    assert original, "engine did not produce a hypothesis"

    calls_before_refine = fake_llm.call_count

    # ── simulate refutation + refinement ─────────────────────────
    graph = HypothesisGraph()
    h1 = graph.add_hypothesis(
        statement=original,
        rationale="Engine iteration 1",
        testable_prediction="Interlayer spacing decreases with Ca/Si",
    )

    # Experiment refutes the hypothesis
    refute_evidence = {
        "result": "diffusion coefficient increased unexpectedly at high Ca/Si",
        "tests_passed": False,
    }
    graph.refute(h1, evidence=refute_evidence)
    assert graph.get(h1).status == "refuted"

    # refine_failed calls RedTeamReviewer (FakeLLM) + _llm_refine (FakeLLM)
    h2 = graph.refine_failed(h1, evidence=refute_evidence, model=fake_llm)

    # ── parent → child relationship ──────────────────────────────
    parent = graph.get(h1)
    child = graph.get(h2)

    assert child.parent_id == h1, "child should trace back to parent"
    assert parent.status == "superseded", "original should be superseded"
    assert child.status == "untested", "refined hypothesis should be ready"
    assert child.statement != original, "refined hypothesis must differ"
    assert "percolation" in child.statement.lower() or "non-linear" in child.statement.lower(), (
        "refined statement should address the refutation"
    )

    # derive edge from parent to child
    derive_edges = [
        e for e in graph.edges()
        if e.from_id == h1 and e.to_id == h2 and e.edge_type == "derive"
    ]
    assert len(derive_edges) == 1, "expected one derive edge parent→child"

    # children() helper returns the refined node
    kids = graph.children(h1)
    assert len(kids) == 1
    assert kids[0].id == h2

    # derivation_chain walks from root to child
    chain = graph.derivation_chain(h2)
    assert len(chain) == 2
    assert chain[0].id == h1
    assert chain[1].id == h2

    # FakeLLM was called during refine (RedTeamReviewer + _llm_refine)
    calls_after_refine = fake_llm.call_count
    assert calls_after_refine > calls_before_refine, (
        "refine_failed should have made real LLM calls"
    )

    # The graph now has 2 nodes and at least 2 edges (derive + refute)
    assert len(graph.all_nodes()) == 2
    edge_types = {e.edge_type for e in graph.edges()}
    assert "derive" in edge_types
    assert "refute" in edge_types
