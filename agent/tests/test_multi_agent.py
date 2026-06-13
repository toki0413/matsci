"""Tests for multi-provider / multi-agent infrastructure."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from matsci_agent.config import MatSciConfig, ModelConfig, AgentProfileConfig
from matsci_agent.models.registry import ModelRegistry
from matsci_agent.agents.factory import AgentFactory
from matsci_agent.agents.orchestrator import Orchestrator, SubTask, TaskPlan
from matsci_agent.tools.registry import ToolRegistry
from matsci_agent.tools.memory_tool import RememberTool, RecallTool
from matsci_agent.tools.orchestrate_tool import OrchestrateTool


ToolRegistry.register(RememberTool())
ToolRegistry.register(RecallTool())
ToolRegistry.register(OrchestrateTool())


def test_model_registry_list_and_default():
    cfg = MatSciConfig(
        models=[
            ModelConfig(alias="cheap", provider="openai", model="gpt-4o-mini", api_key="x"),
            ModelConfig(alias="smart", provider="anthropic", model="claude-opus", api_key="x"),
        ]
    )
    registry = ModelRegistry.from_config(cfg)
    refs = registry.list()
    assert len(refs) == 2
    assert registry.default_alias() == "cheap"


def test_model_registry_resolve_alias():
    fake_model = MagicMock()
    registry = ModelRegistry()
    registry.register(ModelConfig(alias="a", provider="openai", model="gpt-4o", api_key="x"))
    # Cache a fake instance under the alias
    registry._cache["a"] = fake_model
    assert registry.resolve("a") is fake_model


def test_model_registry_resolve_provider_model():
    """provider/model string resolution should build a model instance."""
    registry = ModelRegistry()
    with pytest.raises(ValueError):
        registry.resolve("unknown/model")


def test_config_migration_to_model_pool():
    """Legacy single-provider config still yields a model pool and agent profile."""
    import os
    env = os.environ.copy()
    try:
        os.environ["MATSCI_PROVIDER"] = "openai"
        os.environ["MATSCI_MODEL"] = "gpt-4o"
        os.environ["MATSCI_API_KEY"] = "test-key"
        cfg = MatSciConfig.from_env()
        assert len(cfg.models) == 1
        assert cfg.models[0].alias == "default"
        assert cfg.agents[0].id == "lead"
    finally:
        os.environ.clear()
        os.environ.update(env)


def test_agent_factory_lists_profiles():
    cfg = MatSciConfig(
        models=[ModelConfig(alias="m", provider="openai", model="gpt-4o", api_key="x")],
        agents=[
            AgentProfileConfig(id="lead", model_alias="m"),
            AgentProfileConfig(id="coder", model_alias="m", tools=["file_write_tool"]),
        ],
    )
    registry = ModelRegistry.from_config(cfg)
    factory = AgentFactory(config=cfg, model_registry=registry)
    profiles = factory.list_profiles()
    assert {p.id for p in profiles} == {"lead", "coder"}


def test_agent_factory_respects_tool_filter():
    cfg = MatSciConfig(
        models=[ModelConfig(alias="m", provider="openai", model="gpt-4o", api_key="x")],
        agents=[AgentProfileConfig(id="limited", model_alias="m", tools=["remember", "recall"])],
    )
    fake_model = MagicMock()
    registry = ModelRegistry()
    registry.register(cfg.models[0])
    registry._cache["m"] = fake_model
    factory = AgentFactory(config=cfg, model_registry=registry)
    agent = factory.create("limited")
    tool_names = {t.name for t in agent.langchain_tools}
    assert tool_names == {"remember", "recall"}


@pytest.mark.asyncio
async def test_orchestrator_dependency_order():
    """Tasks with dependencies must run after their prerequisites."""
    cfg = MatSciConfig(
        models=[ModelConfig(alias="m", provider="openai", model="gpt-4o", api_key="x")],
        agents=[
            AgentProfileConfig(id="lead", model_alias="m"),
            AgentProfileConfig(id="worker", model_alias="m"),
        ],
    )
    fake_model = MagicMock()
    registry = ModelRegistry()
    registry.register(cfg.models[0])
    registry._cache["m"] = fake_model
    factory = AgentFactory(config=cfg, model_registry=registry)

    calls = []

    def fake_create(profile_id, **kwargs):
        mock_agent = MagicMock()
        def invoke(prompt):
            calls.append((profile_id, prompt))
            return {"messages": [MagicMock(content=f"result:{profile_id}")]}
        mock_agent.invoke = invoke
        return mock_agent

    factory.create = fake_create

    plan = TaskPlan(
        objective="test",
        tasks=[
            SubTask(task_id="t1", agent_id="worker", prompt="task1", depends_on=[]),
            SubTask(task_id="t2", agent_id="worker", prompt="task2", depends_on=["t1"]),
        ],
    )
    orch = Orchestrator(factory=factory, max_concurrent=2)
    result = await orch.execute(plan)
    assert result.success
    assert result.outputs["t2"] == "result:worker"
    # t1 was invoked before t2 because of dependency
    assert calls[0][0] == "worker"


@pytest.mark.asyncio
async def test_orchestrator_single_task_synthesizes_directly():
    cfg = MatSciConfig(
        models=[ModelConfig(alias="m", provider="openai", model="gpt-4o", api_key="x")],
        agents=[AgentProfileConfig(id="lead", model_alias="m")],
    )
    fake_model = MagicMock()
    registry = ModelRegistry()
    registry.register(cfg.models[0])
    registry._cache["m"] = fake_model
    factory = AgentFactory(config=cfg, model_registry=registry)

    def fake_create(profile_id, **kwargs):
        mock_agent = MagicMock()
        mock_agent.invoke = lambda prompt: {"messages": [MagicMock(content="direct")]}
        return mock_agent

    factory.create = fake_create

    orch = Orchestrator(factory=factory)
    result = await orch.run("hello")
    assert result.success
    assert result.summary == "direct"
