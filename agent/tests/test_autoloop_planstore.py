"""Tests for AutoloopEngine <-> PlanStore integration.

Locks the wiring added when _plan() started persisting plans to PlanStore
and running the confirm/reject gate, plus the complete_plan call after
execute and the store_plan_progress call in _learn.

去 mixin 后 LLM 决策路径不再 mock: 引擎经真实 FakeLLM (BaseChatModel) + 真实
PlanStore + 真实 MemoryManager 驱动; 用户确认门用真实 async 函数模拟 (非
unittest.mock). 仅对 __init__ 里未被测试路径使用的重构造子 (CoderRunner /
ProjectKnowledgeGraph / BenchmarkRunner) 保留 MagicMock 桩 — 与 test_verify_4flags
/ test_autoloop_e2e 一致. 被测逻辑 (plan 解析 / PlanStore 落盘 / 进度记忆) 全真实.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from tests.fixtures.fake_llm import make_callable_llm
from huginn.autoloop.engine import AutoloopEngine
from huginn.autoloop.plan_store import PlanStore
from huginn.memory.manager import MemoryConfig, MemoryManager

_skip_ci_run_cognitive = os.environ.get("HUGINN_CI", "").lower() in ("1", "true", "yes")


class _DummyTracker:
    """Minimal ProgressTracker stand-in -- just absorbs calls."""

    def start_task(self, *a, **kw) -> None: ...
    def update(self, *a, **kw) -> None: ...
    def complete(self, *a, **kw) -> None: ...
    def fail(self, *a, **kw) -> None: ...


class _StubBenchReport:
    passed = 2
    failed = 0
    skipped = 0


class _StubBenchRunner:
    def run(self, categories=None):
        return _StubBenchReport()


class _StubKG:
    """ProjectKnowledgeGraph 轻量桩 — 本组测试不触达 KG."""

    def __init__(self, *a, **kw):
        pass


class _StubCoder:
    """CoderRunner 轻量桩 — 本组测试不触达 coder."""

    def __init__(self, *a, **kw):
        pass


# ── 真实用户确认门 (非 mock) ──────────────────────────────────


async def _auto_confirm(self, *a, **kw) -> None:
    """非昂贵 plan / 无需澄清 → 直接放行."""
    return None


async def _user_agrees(self, *a, **kw) -> bool:
    return True


async def _user_declines(self, *a, **kw) -> bool:
    return False


def _set_plan_response(engine: AutoloopEngine, text: str) -> None:
    """把引擎的真实 FakeLLM 切换到返回指定 plan 文本."""
    object.__setattr__(engine.model, "_func", lambda prompt: text)


def _stub_heavy_constructors(monkeypatch: pytest.MonkeyPatch, fake) -> None:
    """Only genuinely heavy / env-dependent pieces — LLM/plan/memory 全真实."""
    monkeypatch.setattr("huginn.autoloop.engine.get_model", lambda settings: fake)
    monkeypatch.setattr("huginn.autoloop.engine.BenchmarkRunner", lambda *a, **kw: _StubBenchRunner())
    monkeypatch.setattr("huginn.autoloop.engine.CoderRunner", lambda *a, **kw: _StubCoder())
    monkeypatch.setattr("huginn.autoloop.engine.ProjectKnowledgeGraph", lambda *a, **kw: _StubKG())
    monkeypatch.setattr(
        "huginn.agents.speculator.on_turn_start",
        lambda *a, **kw: {"hint": "", "predictions": []},
        raising=False,
    )
    # KB 冷启动跑 ONNX 嵌入 > 120s, KG 写 ~/.huginn — 测试路径不触达, 置 None.
    monkeypatch.setattr("huginn.autoloop.engine.AutoloopEngine._get_kb", lambda self: None)
    monkeypatch.setattr("huginn.autoloop.conjecture.get_kg", lambda *a, **kw: None)


@pytest.fixture
def engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AutoloopEngine:
    """真实 engine: FakeLLM 驱动 LLM, 真实 PlanStore, 真实 MemoryManager."""
    fake = make_callable_llm(
        lambda prompt: "MODE: coder\nDESCRIPTION: tweak x", name="planstore-llm"
    )
    _stub_heavy_constructors(monkeypatch, fake)
    eng = AutoloopEngine(
        workspace=tmp_path,
        memory_manager=MemoryManager(config=MemoryConfig(memory_dir=tmp_path)),
    )
    # 中性化 config 派生的 model_router: 避免 _llm_chat(task=planning) 路由到
    # 真实模型 (会 OpenAIConnectionError / _plan 返 None). 关掉后走 self.model=FakeLLM.
    eng.model_router = None
    eng.progress_tracker = _DummyTracker()
    eng._plan_store = PlanStore(path=tmp_path / "plans.json")
    monkeypatch.setattr(
        "huginn.autoloop.engine.AutoloopEngine._maybe_clarify", _auto_confirm
    )
    return eng


# ── _plan persists to PlanStore ─────────────────────────────────


class TestPlanCreatesPlanstoreEntry:
    def test_plan_creates_planstore_entry(self, engine: AutoloopEngine):
        _set_plan_response(engine, "MODE: coder\nDESCRIPTION: tweak x")

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
    def test_plan_confirmed_when_user_agrees(
        self, engine: AutoloopEngine, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(
            "huginn.autoloop.engine.AutoloopEngine._maybe_clarify", _user_agrees
        )
        _set_plan_response(engine, "MODE: workflow\nDESCRIPTION: run DFT relax")

        plan = asyncio.run(engine._plan("relax GaN", {}))

        assert plan is not None
        persisted = engine._plan_store.get_plan(plan["plan_id"])
        assert persisted is not None
        assert persisted.status == "executing"
        assert persisted.confirmed_at is not None


class TestPlanRejectedWhenUserDeclines:
    def test_plan_rejected_when_user_declines(
        self, engine: AutoloopEngine, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(
            "huginn.autoloop.engine.AutoloopEngine._maybe_clarify", _user_declines
        )
        _set_plan_response(engine, "MODE: workflow\nDESCRIPTION: run DFT relax")

        plan = asyncio.run(engine._plan("relax GaN", {}))

        assert plan is None  # rejected -> _plan returns None
        plans = engine._plan_store.list_plans()
        assert len(plans) == 1
        assert plans[0].status == "abandoned"
        assert plans[0].reject_reason == "user declined"


# ── full run() marks plan complete after execute ────────────────


def _make_stage_llm():
    """Full-cycle callable FakeLLM — hypothesize/plan/reviewer/report 分派."""
    def respond(prompt: str) -> str:
        low = prompt.lower()

        if "testable hypothesis" in low:
            return (
                "If the Ca/Si ratio in C-S-H increases, then the "
                "interlayer spacing decreases, accelerating water "
                "diffusion through the gel pores."
            )

        if "choose one mode" in low:
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
                "water diffusion. Validation confirmed the approach."
            )

        if "修正假设" in prompt or "refined hypothesis" in low:
            return "Refined: percolation-threshold model applies at high Ca/Si."

        return "[]"

    return make_callable_llm(respond, name="planstore-stage-llm")


@pytest.mark.skipif(_skip_ci_run_cognitive, reason="run_cognitive hangs on CI asyncio.run")
class TestPlanCompletedAfterExecute:
    @pytest.mark.asyncio
    async def test_plan_completed_after_execute(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """真实 run_cognitive: 全阶段走 FakeLLM, execute 后 plan 标 completed."""
        from huginn.autoloop.phase_gate import (
            get_shared_phase_gate_state,
            set_shared_phase_gate_state,
        )

        fake_llm = _make_stage_llm()
        _stub_heavy_constructors(monkeypatch, fake_llm)
        engine = AutoloopEngine(
            workspace=tmp_path,
            verification_model=fake_llm,
            memory_manager=MemoryManager(config=MemoryConfig(memory_dir=tmp_path)),
        )
        engine.model_router = None
        engine.progress_tracker = _DummyTracker()
        engine._use_llm_decider = False  # 规则版 decider, FakeLLM 只喂阶段输出
        engine._plan_store = PlanStore(path=tmp_path / "plans.json")
        engine._perceive = lambda: {
            "changed_files": ["diffusion_analysis.py"],
            "git_diff": "+def calc_diffusion(ca_si_ratio): ...",
            "timestamp": "2026-07-04T10:00:00Z",
            "goal": "Optimize C-S-H defect kinetics",
        }
        monkeypatch.setattr(
            "huginn.autoloop.engine.AutoloopEngine._maybe_clarify", _auto_confirm
        )
        (tmp_path / "test_smoke.py").write_text(
            "def test_ok():\n    assert True\n", encoding="utf-8"
        )
        state = get_shared_phase_gate_state()
        state.overrides.add(("validate", "learn"))
        try:
            result = await engine.run_cognitive(
                objective="o", max_iterations=4, progressive_budget=False
            )
        finally:
            state.overrides.discard(("validate", "learn"))
            set_shared_phase_gate_state(None)

        assert result.success is True
        plans = engine._plan_store.list_plans()
        assert len(plans) == 1
        assert plans[0].status == "completed"
        assert plans[0].completed_at is not None


# ── _learn stores plan progress to memory ────────────────────────


class TestPlanProgressStoredInMemory:
    def test_plan_progress_stored_in_memory(self, engine: AutoloopEngine):
        # create a plan via _plan so we get a real plan_id in the real store
        _set_plan_response(engine, "MODE: coder\nDESCRIPTION: do x")

        plan = asyncio.run(engine._plan("some hypothesis", {}))
        assert plan is not None
        assert "plan_id" in plan

        # run the real _learn — 它应经 store_plan_progress 写真实 long-term memory
        asyncio.run(
            engine._learn("some hypothesis", plan, {"tests_passed": True})
        )

        active = engine.memory.load_active_plan()
        assert active is not None, "_learn 应在真实 memory 里落一条 plan 进度"
        assert active["plan_id"] == plan["plan_id"]
        assert active["status"] == "executing"