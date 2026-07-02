"""Orchestrator plan storage 集成测试 — planner/executor 分离 + PlanStore 持久化.

用 stub factory + stub model 隔离 LLM, 只测编排逻辑.
覆盖: plan 持久化 / auto_confirm / review 流 / execute by id /
      draft 拒绝 / 失败标记 / step 状态同步 / 向后兼容.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from huginn.agents.factory import AgentFactory
from huginn.agents.orchestrator import Orchestrator, OrchestratorResult, SubTask, TaskPlan
from huginn.autoloop.plan_store import Plan, PlanStore
from huginn.config import AgentProfileConfig, HuginnConfig, ModelConfig
from huginn.models.registry import ModelRegistry


# ── stub 工厂 ──────────────────────────────────────────────────────────────────


def _make_factory(planner_output: str = '{"steps": [{"id": "s1", "description": "do X"}]}',
                  executor_output: str = "executed result") -> AgentFactory:
    """造一个 stub factory: create_lead 返回 planner mock, create 返回 executor mock."""
    cfg = HuginnConfig(
        models=[ModelConfig(alias="m", provider="openai", model="gpt-4o", api_key="x")],
        agents=[AgentProfileConfig(id="lead", model_alias="m")],
    )
    fake_model = MagicMock()
    registry = ModelRegistry()
    registry.register(cfg.models[0])
    registry._cache["m"] = fake_model
    factory = AgentFactory(config=cfg, model_registry=registry)

    def fake_create_lead(**kwargs):
        mock_agent = MagicMock()
        mock_agent.invoke = lambda prompt: {"messages": [MagicMock(content=planner_output)]}
        return mock_agent

    def fake_create(profile_id, **kwargs):
        mock_agent = MagicMock()
        mock_agent.invoke = lambda prompt: {"messages": [MagicMock(content=executor_output)]}
        return mock_agent

    factory.create_lead = fake_create_lead
    factory.create = fake_create
    return factory


def _make_orchestrator(tmp_path, planner_output=..., executor_output="ok", auto_confirm=False) -> Orchestrator:
    store = PlanStore(path=tmp_path / "plans.json")
    factory = _make_factory(
        planner_output=planner_output if planner_output is not ... else '{"steps": [{"id": "s1", "description": "do X"}]}',
        executor_output=executor_output,
    )
    return Orchestrator(factory=factory, plan_store=store, auto_confirm=auto_confirm)


# ── plan() 持久化 ─────────────────────────────────────────────────────────────


class TestPlanPersist:
    def test_plan_persists_to_store(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        plan = asyncio.run(orch.plan("objective"))
        assert plan.id.startswith("plan_")
        assert plan.status == "draft"
        assert len(plan.steps) == 1
        assert plan.steps[0].id == "s1"
        # store 里有这条记录
        fetched = orch.plan_store.get_plan(plan.id)
        assert fetched is not None
        assert fetched.objective == "objective"

    def test_plan_auto_confirm(self, tmp_path):
        orch = _make_orchestrator(tmp_path, auto_confirm=True)
        plan = asyncio.run(orch.plan("obj"))
        assert plan.status == "confirmed"
        assert plan.confirmed_at is not None

    def test_plan_auto_confirm_per_call(self, tmp_path):
        orch = _make_orchestrator(tmp_path, auto_confirm=False)
        plan = asyncio.run(orch.plan("obj", auto_confirm=True))
        assert plan.status == "confirmed"

    def test_plan_manual_review(self, tmp_path):
        orch = _make_orchestrator(tmp_path, auto_confirm=False)
        plan = asyncio.run(orch.plan("obj"))
        assert plan.status == "draft"
        assert plan.confirmed_at is None

    def test_plan_fallback_on_bad_json(self, tmp_path):
        orch = _make_orchestrator(tmp_path, planner_output="not json at all")
        plan = asyncio.run(orch.plan("fallback obj"))
        assert len(plan.steps) == 1
        assert plan.steps[0].description == "not json at all"


# ── execute() by plan_id ──────────────────────────────────────────────────────


class TestExecuteById:
    def test_execute_confirmed_plan(self, tmp_path):
        orch = _make_orchestrator(tmp_path, executor_output="done output")
        plan = asyncio.run(orch.plan("obj", auto_confirm=True))
        result = asyncio.run(orch.execute(plan.id))
        assert result.success
        assert "done output" in result.outputs.get("s1", "")
        # store 里状态变成 completed
        fetched = orch.plan_store.get_plan(plan.id)
        assert fetched.status == "completed"

    def test_execute_rejects_draft(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        plan = asyncio.run(orch.plan("obj"))  # draft
        with pytest.raises(ValueError, match="need 'confirmed'"):
            asyncio.run(orch.execute(plan.id))

    def test_execute_missing_plan(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        with pytest.raises(ValueError, match="plan not found"):
            asyncio.run(orch.execute("plan_nope"))

    def test_execute_marks_failed_on_error(self, tmp_path):
        """executor agent 抛异常 → plan 状态变 failed."""
        cfg = HuginnConfig(
            models=[ModelConfig(alias="m", provider="openai", model="gpt-4o", api_key="x")],
            agents=[AgentProfileConfig(id="lead", model_alias="m")],
        )
        fake_model = MagicMock()
        registry = ModelRegistry()
        registry.register(cfg.models[0])
        registry._cache["m"] = fake_model
        factory = AgentFactory(config=cfg, model_registry=registry)

        def fake_create_lead(**kwargs):
            mock_agent = MagicMock()
            mock_agent.invoke = lambda p: {"messages": [MagicMock(content='{"steps": [{"id": "s1", "description": "x"}]}')]}
            return mock_agent

        def fake_create(profile_id, **kwargs):
            mock_agent = MagicMock()
            def raise_invoke(prompt):
                raise RuntimeError("executor crashed")
            mock_agent.invoke = raise_invoke
            return mock_agent

        factory.create_lead = fake_create_lead
        factory.create = fake_create

        store = PlanStore(path=tmp_path / "plans.json")
        orch = Orchestrator(factory=factory, plan_store=store, auto_confirm=True)
        plan = asyncio.run(orch.plan("obj"))
        result = asyncio.run(orch.execute(plan.id))
        assert not result.success
        fetched = orch.plan_store.get_plan(plan.id)
        assert fetched.status == "failed"

    def test_step_statuses_updated_after_execute(self, tmp_path):
        orch = _make_orchestrator(tmp_path, executor_output="ok")
        plan = asyncio.run(orch.plan("obj", auto_confirm=True))
        asyncio.run(orch.execute(plan.id))
        fetched = orch.plan_store.get_plan(plan.id)
        step = fetched.steps[0]
        assert step.status == "done"
        assert step.result == "ok"


# ── run() 分支 ────────────────────────────────────────────────────────────────


class TestRunBranches:
    def test_run_auto_confirm_executes(self, tmp_path):
        orch = _make_orchestrator(tmp_path, executor_output="final answer", auto_confirm=True)
        result = asyncio.run(orch.run("obj"))
        assert isinstance(result, OrchestratorResult)
        assert result.success

    def test_run_review_returns_draft_plan(self, tmp_path):
        orch = _make_orchestrator(tmp_path, auto_confirm=False)
        result = asyncio.run(orch.run("obj"))
        assert isinstance(result, Plan)
        assert result.status == "draft"


# ── 向后兼容 ──────────────────────────────────────────────────────────────────


class TestBackwardCompat:
    def test_execute_taskplan_without_store(self, tmp_path):
        """plan_store=None 时 execute(TaskPlan) 走老路径."""
        factory = _make_factory(executor_output="legacy ok")
        orch = Orchestrator(factory=factory)  # plan_store=None
        tp = TaskPlan(
            objective="legacy",
            tasks=[SubTask(task_id="t1", agent_id="lead", prompt="do it")],
        )
        result = asyncio.run(orch.execute(tp))
        assert result.success
        assert "legacy ok" in result.outputs.get("t1", "")

    def test_run_without_store_legacy_path(self, tmp_path):
        """plan_store=None 时 run() 走老的 plan→execute→synthesize 路径."""
        factory = _make_factory(
            planner_output='{"tasks": [{"task_id": "t1", "agent_id": "lead", "prompt": "x"}]}',
            executor_output="legacy run",
        )
        orch = Orchestrator(factory=factory)  # plan_store=None
        result = asyncio.run(orch.run("obj"))
        assert isinstance(result, OrchestratorResult)
        assert result.success
        assert result.summary == "legacy run"  # 单 task 直接返回结果


# ── review 全流程 ─────────────────────────────────────────────────────────────


class TestReviewFlow:
    def test_plan_confirm_execute_flow(self, tmp_path):
        """完整 review 流: plan (draft) → confirm → execute → completed."""
        orch = _make_orchestrator(tmp_path, executor_output="review ok")
        # 1. plan → draft
        plan = asyncio.run(orch.plan("review obj"))
        assert plan.status == "draft"
        # 2. user confirms
        orch.plan_store.confirm_plan(plan.id)
        # 3. execute
        result = asyncio.run(orch.execute(plan.id))
        assert result.success
        # 4. store 状态
        fetched = orch.plan_store.get_plan(plan.id)
        assert fetched.status == "completed"
        assert fetched.steps[0].status == "done"
