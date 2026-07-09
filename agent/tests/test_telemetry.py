"""Tests for the telemetry collector."""

from __future__ import annotations

import asyncio
import time

from huginn.agent import HuginnAgent
from huginn.telemetry import (
    TelemetryCollector,
    set_telemetry_collector,
)


class TestTelemetryCollector:
    def test_span_records_duration(self):
        collector = TelemetryCollector()
        with collector.span("test") as span:
            time.sleep(0.01)
        assert span.duration_ms >= 10
        assert span.end_time is not None

    def test_nested_spans(self):
        collector = TelemetryCollector()
        with collector.span("outer") as outer, collector.span("inner"):
            pass
        assert len(outer.children) == 1
        assert outer.children[0].name == "inner"

    def test_summary_aggregates(self):
        collector = TelemetryCollector()
        with collector.span("a"):
            pass
        with collector.span("a"):
            pass
        summary = collector.summary()
        assert summary["by_name"]["a"]["count"] == 2

    def test_clear(self):
        collector = TelemetryCollector()
        with collector.span("x"):
            pass
        assert len(collector.to_dict()) == 1
        collector.clear()
        assert len(collector.to_dict()) == 0


class TestAgentTelemetry:
    def test_agent_turn_span(self):
        class _FakeGraph:
            async def astream(self, inputs, config, stream_mode):
                yield ("values", {"messages": []})

        set_telemetry_collector(TelemetryCollector())
        agent = HuginnAgent(model=None, tools=[])
        agent._agent_graph = _FakeGraph()

        asyncio.run(_consume(agent, "hello"))

        summary = agent.telemetry_summary()
        assert "agent_turn" in summary["by_name"]


async def _consume(agent: HuginnAgent, message: str) -> None:
    async for _ in agent.chat(message):
        pass
