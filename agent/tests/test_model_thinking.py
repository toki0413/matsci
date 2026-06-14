"""Tests for model thinking/reasoning intensity configuration."""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

from huginn.agents.factory import AgentFactory
from huginn.config import HuginnConfig, ModelConfig, AgentProfileConfig
from huginn.models.registry import ModelRegistry, create_langchain_model


class _FakeChatModel:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


def _patch_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "langchain_anthropic.ChatAnthropic",
        _FakeChatModel,
    )


def _patch_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "langchain_openai.ChatOpenAI",
        _FakeChatModel,
    )


class TestCreateLangchainModelThinking:
    def test_anthropic_low_intensity(self, monkeypatch: pytest.MonkeyPatch):
        _patch_anthropic(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        model = create_langchain_model(
            provider="anthropic",
            model_name="claude-sonnet-4",
            thinking="low",
        )
        assert model.kwargs["thinking"] == {"type": "enabled", "budget_tokens": 4096}
        assert model.kwargs["max_tokens"] == 8192

    def test_anthropic_high_intensity_respects_max_tokens(self, monkeypatch: pytest.MonkeyPatch):
        _patch_anthropic(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        model = create_langchain_model(
            provider="anthropic",
            model_name="claude-sonnet-4",
            thinking="high",
            max_tokens=50000,
        )
        assert model.kwargs["thinking"] == {"type": "enabled", "budget_tokens": 32000}
        assert model.kwargs["max_tokens"] == 50000

    def test_anthropic_dict_passthrough(self, monkeypatch: pytest.MonkeyPatch):
        _patch_anthropic(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        custom = {"type": "enabled", "budget_tokens": 12345}
        model = create_langchain_model(
            provider="anthropic",
            model_name="claude-sonnet-4",
            thinking=custom,
            max_tokens=20000,
        )
        assert model.kwargs["thinking"] == custom
        assert model.kwargs["max_tokens"] == 20000

    def test_openai_reasoning_effort_for_o_series(self, monkeypatch: pytest.MonkeyPatch):
        _patch_openai(monkeypatch)
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        model = create_langchain_model(
            provider="openai",
            model_name="o3-mini",
            thinking="medium",
        )
        assert model.kwargs["reasoning_effort"] == "medium"

    def test_openai_no_reasoning_effort_for_regular_model(self, monkeypatch: pytest.MonkeyPatch):
        _patch_openai(monkeypatch)
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        model = create_langchain_model(
            provider="openai",
            model_name="gpt-4o",
            thinking="medium",
        )
        assert "reasoning_effort" not in model.kwargs

    def test_no_thinking_by_default(self, monkeypatch: pytest.MonkeyPatch):
        _patch_openai(monkeypatch)
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        model = create_langchain_model(
            provider="openai",
            model_name="gpt-4o",
        )
        assert "reasoning_effort" not in model.kwargs


class TestConfigParsing:
    def test_models_json_with_thinking(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv(
            "HUGINN_MODELS",
            json.dumps([
                {
                    "alias": "claude",
                    "provider": "anthropic",
                    "model": "claude-sonnet-4",
                    "thinking": "high",
                    "max_tokens": 48000,
                }
            ]),
        )
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        cfg = HuginnConfig.from_env()
        assert cfg.models[0].thinking == "high"
        assert cfg.models[0].max_tokens == 48000

    def test_global_thinking_fallback(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("HUGINN_PROVIDER", "openai")
        monkeypatch.setenv("HUGINN_MODEL", "o1")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("HUGINN_THINKING", "low")
        cfg = HuginnConfig.from_env()
        assert cfg.thinking == "low"
        assert cfg.models[0].thinking == "low"

    def test_global_thinking_dict_fallback(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("HUGINN_PROVIDER", "anthropic")
        monkeypatch.setenv("HUGINN_MODEL", "claude-sonnet-4")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setenv("HUGINN_THINKING", json.dumps({"type": "enabled", "budget_tokens": 8000}))
        cfg = HuginnConfig.from_env()
        assert cfg.thinking == {"type": "enabled", "budget_tokens": 8000}


class TestModelRegistryOverride:
    def test_request_level_override_creates_separate_cache_entry(self, monkeypatch: pytest.MonkeyPatch):
        _patch_openai(monkeypatch)
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        registry = ModelRegistry(models=[
            ModelConfig(alias="o1", provider="openai", model="o1", thinking="low"),
        ])
        default_model = registry.get("o1")
        override_model = registry.get("o1", thinking="high")
        assert default_model is not override_model
        assert default_model.kwargs["reasoning_effort"] == "low"
        assert override_model.kwargs["reasoning_effort"] == "high"


class TestAgentFactoryOverride:
    def test_factory_passes_profile_thinking(self, monkeypatch: pytest.MonkeyPatch):
        _patch_anthropic(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        config = HuginnConfig(
            models=[ModelConfig(alias="claude", provider="anthropic", model="claude-sonnet-4")],
            agents=[AgentProfileConfig(id="lead", model_alias="claude", thinking="medium")],
        )
        factory = AgentFactory(config)
        agent = factory.create("lead")
        assert agent.model.kwargs["thinking"] == {"type": "enabled", "budget_tokens": 16000}

    def test_factory_request_override(self, monkeypatch: pytest.MonkeyPatch):
        _patch_anthropic(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        config = HuginnConfig(
            models=[ModelConfig(alias="claude", provider="anthropic", model="claude-sonnet-4")],
            agents=[AgentProfileConfig(id="lead", model_alias="claude")],
        )
        factory = AgentFactory(config)
        agent = factory.create("lead", thinking="high")
        assert agent.model.kwargs["thinking"] == {"type": "enabled", "budget_tokens": 32000}
