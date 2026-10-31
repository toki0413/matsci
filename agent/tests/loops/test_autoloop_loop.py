"""P0 integration tests for the autoloop engine (autoloop/engine.py).

Drives the real AutoloopEngine.run_cognitive() end-to-end with a FakeLLM in
callable mode. Heavy external calls (BenchmarkRunner, CoderRunner,
PerceptionLayer) are stubbed — the LLM decision path is never mocked.

Test matrix:
  1. Full 7-phase pipeline completes
  2. 空 description 在 plan 内部重试, 合法 plan 过 phase gate
  3. Iteration budget (max_iterations) limits the loop
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

from huginn.autoloop.engine import AutoloopEngine
from huginn.autoloop.phase_gate import get_shared_phase_gate_state
from huginn.autoloop.types import AutoloopResult
from huginn.memory.manager import MemoryManager
from tests.fixtures.fake_llm import make_callable_llm

# Full-cycle autoloop 跑 7 个 phase, 本地 ~230s. CI runner --timeout=120 会杀.
# 跟 test_chat_loop 同款: HUGINN_CI=1 时 skip, 本地照跑.
_skip_ci = os.environ.get("HUGINN_CI", "").lower() in ("1", "true", "yes")
pytestmark = pytest.mark.skipif(
    _skip_ci, reason="autoloop full cycle too slow on CI runner"
)

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


async def _noop_maybe_clarify(self, *a, **kw):
    """Async no-op for _maybe_clarify — avoids ClarificationManager.ask
    dead-waiting (asyncio.wait_for(future, 60)) on a Future that never
    gets resolve()'d in the test environment."""
    return None


def _stub_heavy_calls(monkeypatch, fake_llm):
    monkeypatch.setattr("huginn.autoloop.engine.get_model", lambda settings: fake_llm)
    monkeypatch.setattr(
        "huginn.autoloop.engine.BenchmarkRunner", lambda: _StubBenchRunner()
    )
    monkeypatch.setattr(
        "huginn.autoloop.engine.CoderRunner", lambda *a, **kw: MagicMock()
    )
    monkeypatch.setattr(
        "huginn.autoloop.engine.ProjectKnowledgeGraph", lambda *a, **kw: MagicMock()
    )
    monkeypatch.setattr(
        "huginn.agents.speculator.on_turn_start",
        lambda *a, **kw: {"hint": "", "predictions": []},
        raising=False,
    )
    # ponytail: KB 第一次懒加载会 seed_knowledge_base 跑 ONNX 嵌入,
    # CI 首次冷启动 > 120s timeout. _build_kb_text 已处理 None, 直接跳过.
    monkeypatch.setattr(
        "huginn.autoloop.engine.AutoloopEngine._get_kb", lambda self: None
    )
    # KG 单例从 conjecture.get_kg 懒加载到 ~/.huginn, CI 上会真写文件污染 home.
    # get_kg 返回 None 时 _persist_to_kg/_fetch_domain_context 都安全跳过.
    monkeypatch.setattr("huginn.autoloop.conjecture.get_kg", lambda *a, **kw: None)
    # _maybe_clarify("hypothesize_align") → ClarificationManager.ask(timeout=60)
    # 死等一个测试环境里永不 resolve() 的 Future. patch 成 async no-op.
    monkeypatch.setattr(AutoloopEngine, "_maybe_clarify", _noop_maybe_clarify)


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
    # v10: 关 LLM decider, 走规则版 (FakeLLM 响应按规则版顺序设计)
    engine._use_llm_decider = False
    # 中性化 config 派生的 model_router (同 test_verify_4flags): 全量套件里
    # 全局 config 缓存可能被喂成含 real provider 的 router, 导致 _llm_chat(
    # task="planning") 路由到真实模型而非本测试注入的 FakeLLM, _plan 返 None
    # → "缺 execute". 本测试意图是 FakeLLM 驱动所有阶段, 必须关掉路由.
    engine.model_router = None
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
            # v10: run_cognitive 1-action-per-iter, max_iter=8 跑完 6 phases + report
            result = await engine.run_cognitive(
                objective="Optimize C-S-H defect kinetics",
                max_iterations=8,
                progressive_budget=False,
            )
        finally:
            _restore_gate(gate_state)

        assert isinstance(result, AutoloopResult)
        names = [p.name for p in result.phases]
        # 6 stages 全跑完 + report
        for expected in (
            "hypothesize",
            "plan",
            "execute",
            "validate",
            "learn",
            "report",
        ):
            assert expected in names, f"缺 {expected}, 实际 {names}"
        assert len(names) >= 7, f"phase 数 < 7, 实际 {names}"
        for phase in result.phases:
            assert phase.status == "completed", (
                f"phase '{phase.name}' status={phase.status}"
            )
        # LLM was actually called (not mocked)
        assert fake_llm.call_count >= 3


# ── 2. phase gate: 空 description → plan 重试 → 合法 plan 过门 ───────


def _gate_retry_plan_fn():
    """First plan 空 description, 重试补全.

    结构性锚点: 空 description 不得作为执行对象下发, `_plan` 内部重试一次;
    门控看到的是补全后的 plan, 不会再出现「空 plan 被 advisory 放行」的 phase.
    """
    plan_calls = [0]

    def plan_fn():
        plan_calls[0] += 1
        if plan_calls[0] == 1:
            return ""
        return "MODE: coder\nDESCRIPTION: Add a convergence check to the script."

    return plan_fn, plan_calls


def _assert_gate_retry_phases(result, plan_calls) -> None:
    # 空描述重试发生在 plan 阶段内部, plan phase 记录的是补全后的 plan.
    plan_phases = [p for p in result.phases if p.name == "plan"]
    names = [p.name for p in result.phases]
    assert len(plan_phases) >= 1, f"缺 plan phase, 实际 {names}"
    assert plan_calls[0] >= 2, f"空描述应触发 plan 内部重试, 实际调用 {plan_calls[0]} 次"
    assert all(p.result["description"] != "" for p in plan_phases), (
        f"空 description 不得作为 plan 结果留下: {[p.result for p in plan_phases]}"
    )

    # 重试补全后流水线继续
    assert any(p.name == "execute" for p in result.phases), names
    assert any(p.name == "report" for p in result.phases), names


class TestAutoloopPhaseGate:
    @pytest.mark.asyncio
    async def test_empty_plan_description_retried_before_gate(
        self, tmp_path, monkeypatch
    ):
        """First plan 空 description → plan 内部重试 → 合法 plan 过门后继续."""
        plan_fn, plan_calls = _gate_retry_plan_fn()
        fake_llm = _make_stage_llm(plan_fn=plan_fn)
        engine, _ = _make_engine(tmp_path, fake_llm, monkeypatch)

        (tmp_path / "test_smoke.py").write_text(
            "def test_ok():\n    assert True\n", encoding="utf-8"
        )
        gate_state = _bypass_validate_gate()
        try:
            # v10: run_cognitive 1-action-per-iter. max_iter=12 让 gate 阻断 +
            # retry 都跑完 (hyp→plan(blocked)→plan(retry)→exec→val→learn + report)
            result = await engine.run_cognitive(
                objective="Optimize C-S-H defect kinetics",
                max_iterations=12,
                progressive_budget=False,
            )
        finally:
            _restore_gate(gate_state)

        _assert_gate_retry_phases(result, plan_calls)


# ── 3. budget exhaustion: max_iterations limit → stops ─────────────


class TestAutoloopBudgetExhaustion:
    @pytest.mark.asyncio
    async def test_max_iterations_limits_loop(self, tmp_path, monkeypatch):
        """max_iterations=2 → 2 actions + report, stops at max_iter."""
        fake_llm = _make_stage_llm()
        engine, _ = _make_engine(tmp_path, fake_llm, monkeypatch)

        (tmp_path / "test_smoke.py").write_text(
            "def test_ok():\n    assert True\n", encoding="utf-8"
        )
        gate_state = _bypass_validate_gate()
        try:
            # v10: run_cognitive 1-action-per-iter, max_iter=2 = 2 actions + report
            result = await engine.run_cognitive(
                objective="Optimize C-S-H defect kinetics",
                max_iterations=2,
                progressive_budget=False,
            )
        finally:
            _restore_gate(gate_state)

        # 2 actions (hypothesize + plan) + report = 3 phases
        names = [p.name for p in result.phases]
        assert "report" in names, f"缺 report, 实际 {names}"
        actions = [n for n in names if n != "report"]
        assert len(actions) == 2, f"max_iter=2 应该 2 actions, 实际 {names}"

    @pytest.mark.asyncio
    async def test_progressive_budget_rejects_expensive_mode(
        self, tmp_path, monkeypatch
    ):
        """Workflow mode is rejected at medium tier (iteration 11+).

        ponytail: We don't run 11 full iterations (too slow). Instead we
        verify the budget mechanism directly: feed a 'workflow' plan at
        iteration 11 and check _check_budget returns False.
        """
        from huginn.autoloop.budget import ProgressiveBudget

        budget = ProgressiveBudget.default()
        engine, _ = _make_engine(tmp_path, _make_stage_llm(), monkeypatch)
        engine._budget = budget
        engine._budget_degraded = False
        engine._budget_rejects = {}

        # Iteration 11 falls in the 'medium' tier where 'workflow' is not allowed
        allowed = engine._check_budget(11, {"mode": "workflow"})
        assert allowed is False, "workflow should be rejected at iteration 11"

        # 'coder' is always allowed
        allowed_coder = engine._check_budget(11, {"mode": "coder"})
        assert allowed_coder is True

        # At iteration 1 (open tier), 'workflow' is allowed
        allowed_early = engine._check_budget(1, {"mode": "workflow"})
        assert allowed_early is True
