"""Tests for HuginnAgent._maybe_inject_proactive_suggestion.

Covers: method existence, empty-pipeline no-crash, and message injection
when the pipeline reports ready suggestions.
"""

import asyncio
from types import MethodType
from unittest.mock import patch

import pytest

from huginn.agent import HuginnAgent
from huginn.provenance.pipeline import PipelineStage, PipelineSuggestion


def _make_suggestion(stage, hint, desc, ready=True):
    return PipelineSuggestion(
        stage=stage,
        tool_hint=hint,
        description=desc,
        prerequisite_met=ready,
        reason="test",
    )


class _FakePipeline:
    """Minimal stand-in for SimulationPipeline."""

    def __init__(self, latest=None, latest_entry=None):
        self._latest = latest or []
        self._entry = latest_entry

    def _latest_entry(self):
        return self._entry

    def suggest_next(self, tool_name, tool_input, tool_output):
        return self._latest


class _FakeAgent:
    """Just enough surface to bind the method onto."""

    def __init__(self):
        self._pending_synthetic_messages = []


def test_method_exists_and_callable():
    assert hasattr(HuginnAgent, "_maybe_inject_proactive_suggestion")
    assert callable(getattr(HuginnAgent, "_maybe_inject_proactive_suggestion"))


def test_empty_pipeline_no_crash():
    agent = _FakeAgent()

    async def run():
        with patch(
            "huginn.provenance.pipeline.get_pipeline",
            return_value=_FakePipeline(latest=[], latest_entry=None),
        ):
            await HuginnAgent._maybe_inject_proactive_suggestion(agent)

    asyncio.run(run())
    # nothing injected when there are no suggestions
    assert agent._pending_synthetic_messages == []


def test_injects_when_ready_suggestions():
    agent = _FakeAgent()
    suggestions = [
        _make_suggestion(PipelineStage.RELAX, "vasp_tool", "结构弛豫完成"),
        _make_suggestion(PipelineStage.PROPERTIES, "band_tool", "计算 band structure"),
    ]

    async def run():
        with patch(
            "huginn.provenance.pipeline.get_pipeline",
            return_value=_FakePipeline(latest=suggestions),
        ):
            await HuginnAgent._maybe_inject_proactive_suggestion(agent)

    asyncio.run(run())
    assert len(agent._pending_synthetic_messages) == 1
    content = agent._pending_synthetic_messages[0].content
    assert "[Pipeline Suggestion]" in content
    assert "band_tool" in content


def test_skips_when_prerequisites_not_met():
    agent = _FakeAgent()
    suggestions = [
        _make_suggestion(PipelineStage.RELAX, "vasp_tool", "弛豫", ready=False),
    ]

    async def run():
        with patch(
            "huginn.provenance.pipeline.get_pipeline",
            return_value=_FakePipeline(latest=suggestions),
        ):
            await HuginnAgent._maybe_inject_proactive_suggestion(agent)

    asyncio.run(run())
    assert agent._pending_synthetic_messages == []
