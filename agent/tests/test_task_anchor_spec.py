"""反跑题任务锚 + 用户指定子代理 (@spec) 的最小检查.

覆盖两处新逻辑:
1. factory.create 在未 override 时注入 # Task Anchor, override 时不注入.
2. SubagentDispatch.dispatch 仅用户显式 @点名 的 spec 才派发 (未知 spec 干净失败).
"""

from __future__ import annotations

import asyncio
import unittest.mock as mock

from huginn.agents.factory import AgentFactory, _TASK_ANCHOR


class _FakePersona:
    system_prompt = "base persona"
    begin_dialogs: list[dict] = []


class _FakeProfile:
    model_alias = "deepseek-v4-flash"
    persona = "researcher"
    thinking = None
    tools: list[str] | None = None


def _min_factory() -> AgentFactory:
    """object.__new__ 绕过 __init__, 手动装配 create 用到的最小字段."""
    factory = object.__new__(AgentFactory)
    factory.config = mock.Mock()
    factory.memory_manager = None
    factory.config.workspace = "/tmp/huginn-test"
    factory.config.privacy_redact_secrets = False
    factory.config.privacy_block_on_secrets = False
    factory.config.max_tool_output_tokens = 4096
    factory.config.context_budget_tokens = 16000
    factory.config.auto_approve = True
    factory.config.tool_compression_max_tokens = 2048
    factory.config.telemetry_enabled = False
    factory.config.memory_decay_enabled = False
    factory.config.memory_decay_interval_turns = 0
    factory.config.memory_decay_prune_threshold = 0
    factory.config.checkpointer_path = None
    factory.config.hpc_scheduler = "local"
    factory.config.hpc_host = ""
    factory.config.hpc_username = ""
    factory._profiles = {"researcher": _FakeProfile()}
    factory.model_registry = mock.Mock()
    factory.model_registry.resolve.return_value = mock.Mock()
    factory.persona_manager = mock.Mock()
    factory.persona_manager.get.return_value = _FakePersona()
    factory._shared_checkpointer = mock.Mock()
    factory._shared_scheduler = None
    factory._anomaly_store = mock.Mock()
    return factory


def test_task_anchor_injected_without_override():
    """未 override 时, create 注入 # Task Anchor 到 system prompt."""
    with mock.patch("huginn.agents.factory.HuginnAgent") as FakeAgent, \
         mock.patch("huginn.agents.factory.EmotionTracker") as FakeEmotion, \
         mock.patch("huginn.agents.factory.load_project_context", return_value=""):
        factory = _min_factory()
        factory.create("researcher", thread_id="t")

    _, kwargs = FakeAgent.call_args
    prompt = kwargs["system_prompt"]
    assert "# Task Anchor" in prompt
    assert _TASK_ANCHOR in prompt
    assert prompt.startswith(_FakePersona.system_prompt + "\n\n# Task Anchor")


def test_task_anchor_skipped_on_override():
    """显式 override 时, 不注入任务锚 — 尊重调用方提供的完整 prompt."""
    with mock.patch("huginn.agents.factory.HuginnAgent") as FakeAgent, \
         mock.patch("huginn.agents.factory.EmotionTracker") as FakeEmotion, \
         mock.patch("huginn.agents.factory.load_project_context", return_value=""):
        factory = _min_factory()
        override = "custom system prompt"
        factory.create("researcher", thread_id="t", system_prompt_override=override)

    _, kwargs = FakeAgent.call_args
    assert kwargs["system_prompt"] == override


def test_spec_dispatch_requires_explicit_token():
    """仅用户 @点名 的 spec 才派发; 未知 spec 干净失败不落副作用."""
    from tests.test_e2e_agent_loop import _FakeFactory, _FakeSubagent
    from huginn.agents.subagent import SubagentDispatch

    factory = _FakeFactory(_FakeSubagent())
    dispatch = SubagentDispatch()

    # 合法 spec 派发成功
    result = asyncio.run(
        dispatch.dispatch("explore", "find DFT entries", context={"agent_factory": factory})
    )
    assert result.success is True
    assert result.spec_name == "explore"

    # 未知 spec 干净失败
    result = asyncio.run(
        dispatch.dispatch("nope", "task", context={"agent_factory": factory})
    )
    assert result.success is False