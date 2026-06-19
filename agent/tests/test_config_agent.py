"""Tests for HuginnConfig.build_agent_kwargs and HuginnAgent.from_config."""

from __future__ import annotations

import pytest
pytest.importorskip("langchain_ollama", reason="langchain-ollama not installed")

import tempfile
from pathlib import Path

from huginn.agent import HuginnAgent
from huginn.config import HuginnConfig, ModelConfig


class TestConfigAgentBuilder:
    def test_legacy_single_model_kwargs(self):
        cfg = HuginnConfig(provider="ollama", model="qwen2.5:14b")
        kwargs = cfg.build_agent_kwargs()
        assert kwargs["model"] is not None
        assert kwargs["model_router"] is None
        assert kwargs["prompt_cache_control"] is True

    def test_model_router_kwargs(self):
        cfg = HuginnConfig(
            models=[
                ModelConfig(alias="a", provider="ollama", model="qwen2.5:14b"),
            ]
        )
        kwargs = cfg.build_agent_kwargs()
        assert kwargs["model_router"] is not None
        assert kwargs["model"] is None

    def test_from_config_in_memory_checkpointer(self):
        cfg = HuginnConfig(provider="ollama", model="qwen2.5:14b")
        agent = HuginnAgent.from_config(cfg)
        assert isinstance(agent, HuginnAgent)
        assert "InMemory" in type(agent.checkpointer).__name__
        agent.close()

    def test_from_config_persistent_checkpointer(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = HuginnConfig(
                provider="ollama",
                model="qwen2.5:14b",
                checkpointer_path=str(Path(tmp) / "cp.sqlite"),
            )
            with HuginnAgent.from_config(cfg) as agent:
                assert "Sqlite" in type(agent.checkpointer).__name__

    def test_telemetry_collector_is_per_agent(self):
        cfg = HuginnConfig(provider="ollama", model="qwen2.5:14b")
        a1 = HuginnAgent.from_config(cfg)
        a2 = HuginnAgent.from_config(cfg)
        assert a1._telemetry_collector is not a2._telemetry_collector
        a1.close()
        a2.close()