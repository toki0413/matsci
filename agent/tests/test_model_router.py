"""Tests for the multi-LLM model router."""

from __future__ import annotations

import pytest

from huginn.agent import HuginnAgent
from huginn.models.router import ModelRouter


class _FakeModel:
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"FakeModel({self.name})"


class TestModelRouter:
    def test_select_by_tag_priority(self):
        router = ModelRouter()
        router.register(
            "cheap", _FakeModel("cheap"), tags={"cheap", "summarize"}, priority=1
        )
        router.register(
            "strong", _FakeModel("strong"), tags={"science", "reasoning"}, priority=5
        )
        router.register("default", _FakeModel("default"), tags={"default"}, priority=0)

        assert router.select("science").name == "strong"
        assert router.select("summarize").name == "cheap"
        assert router.select("unknown").name == "default"

    def test_prefer_cheap(self):
        router = ModelRouter()
        router.register(
            "a", _FakeModel("a"), tags={"default"}, cost_input=10, priority=5
        )
        router.register(
            "b", _FakeModel("b"), tags={"default"}, cost_input=1, priority=1
        )

        assert router.select("default").name == "a"
        assert router.select("default", prefer_cheap=True).name == "b"

    def test_empty_router_raises(self):
        with pytest.raises(RuntimeError):
            ModelRouter().select("default")

    def test_from_env_skips_invalid(self, monkeypatch):
        monkeypatch.setenv("HUGINN_MODEL_CHEAP", "openai:gpt-4o-mini")
        # No OPENAI_API_KEY, so registration should be skipped.
        router = ModelRouter.from_env()
        assert "cheap" not in router.list_models()


class TestHuginnAgentRouter:
    def test_select_model_uses_router(self):
        router = ModelRouter()
        router.register("m", _FakeModel("routed"), tags={"agent"})
        agent = HuginnAgent(model_router=router)
        assert agent.select_model("agent").name == "routed"

    def test_select_model_falls_back_to_single_model(self):
        model = _FakeModel("single")
        agent = HuginnAgent(model=model)
        assert agent.select_model("agent").name == "single"

    def test_no_model_raises(self):
        agent = HuginnAgent()
        with pytest.raises(RuntimeError):
            agent.select_model("agent")
