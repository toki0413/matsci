"""Tests for local-only / no-cloud mode."""

from __future__ import annotations

import contextlib

import pytest

from huginn.config import HuginnConfig, ModelConfig
from huginn.models.registry import ModelRegistry, is_local_provider


def test_is_local_provider() -> None:
    assert is_local_provider("ollama") is True
    assert is_local_provider("vllm", "http://localhost:8000") is True
    assert is_local_provider("local", "http://127.0.0.1:8000") is True
    assert is_local_provider("openai", "http://localhost:8000") is True
    assert is_local_provider("openai") is False
    assert is_local_provider("anthropic") is False


def test_local_only_blocks_cloud_provider() -> None:
    config = HuginnConfig(
        local_only_mode=True,
        models=[ModelConfig(alias="claude", provider="anthropic", model="claude-3")],
    )
    registry = ModelRegistry.from_config(config)
    with pytest.raises(ValueError, match="Local-only mode"):
        registry.get("claude")


def test_local_only_allows_ollama() -> None:
    config = HuginnConfig(
        local_only_mode=True,
        models=[ModelConfig(alias="local", provider="ollama", model="qwen2.5:14b")],
    )
    registry = ModelRegistry.from_config(config)
    # Ollama import may not be installed, but the local-only check should pass first.
    # We just verify no ValueError about local-only mode is raised.
    with contextlib.suppress(ImportError):
        registry.get("local")
