"""Tests that high-impact config flags are actually wired into the runtime."""

from __future__ import annotations

import pytest
pytest.importorskip("langchain_ollama", reason="langchain-ollama not installed")

from huginn.agent import HuginnAgent
from huginn.config import HuginnConfig
from huginn.telemetry import NullTelemetryCollector


class TestConfigFlagsWired:
    def test_auto_approve_flag(self):
        cfg = HuginnConfig(provider="ollama", model="qwen2.5:14b", auto_approve=True)
        agent = HuginnAgent.from_config(cfg)
        assert agent.auto_approve is True
        assert agent._permission_config.auto_approve_all is True
        agent.close()

    def test_tool_compression_max_tokens(self):
        cfg = HuginnConfig(
            provider="ollama",
            model="qwen2.5:14b",
            tool_compression_max_tokens=4096,
        )
        agent = HuginnAgent.from_config(cfg)
        assert agent.compression_max_tokens == 4096
        agent.close()

    def test_telemetry_disabled_uses_null_collector(self):
        cfg = HuginnConfig(
            provider="ollama",
            model="qwen2.5:14b",
            telemetry_enabled=False,
        )
        agent = HuginnAgent.from_config(cfg)
        assert isinstance(agent._telemetry_collector, NullTelemetryCollector)
        agent.close()

    def test_telemetry_enabled_uses_real_collector(self):
        cfg = HuginnConfig(
            provider="ollama",
            model="qwen2.5:14b",
            telemetry_enabled=True,
        )
        agent = HuginnAgent.from_config(cfg)
        assert type(agent._telemetry_collector).__name__ == "TelemetryCollector"
        agent.close()

    def test_memory_decay_flags(self):
        cfg = HuginnConfig(
            provider="ollama",
            model="qwen2.5:14b",
            memory_decay_enabled=True,
            memory_decay_interval_turns=3,
            memory_decay_prune_threshold=0.2,
        )
        agent = HuginnAgent.from_config(cfg)
        assert agent.memory_decay_enabled is True
        assert agent.memory_decay_interval_turns == 3
        assert agent.memory_decay_prune_threshold == 0.2
        agent.close()