"""Tests for the communication contract (ASD-STE100 discipline, structured layer).

Covers the deterministic contract module, the advisory audit wired into the
event bus, and the ``comms_lint`` tool.
"""

import time

from huginn.comms.contract import (
    KNOWN_EVENT_TYPES,
    audit_agent_event,
    audit_agent_message,
    audit_subagent_result,
    lint,
)
from huginn.core_types import AgentMessage, ToolContext
from huginn.events.event_bus import AgentEvent, EventBus
from huginn.plugins.asd_ste100.main import (
    COMMS_TOOL_NAME,
    SEGMENT_NAME,
    TOOL_NAME,
    CommsLintInput,
    CommsLintTool,
)
from huginn.plugins.prompt_segments import registered_prompt_segments
from huginn.tools.registry import ToolRegistry


def _ctx() -> ToolContext:
    return ToolContext(session_id="t", workspace="/tmp")


def _rules(report) -> set[str]:
    return {v.rule for v in report.violations}


class TestVocabulary:
    def test_declared_events_present_and_wildcard_excluded(self):
        assert "tool.result" in KNOWN_EVENT_TYPES
        assert "*" not in KNOWN_EVENT_TYPES

    def test_emitted_but_undeclared_stay_advisory(self):
        # These are genuinely published in the repo but deliberately left out of
        # the non-exhaustive ``ALL_TYPES`` list.  The contract surfaces them as
        # an advisory ``type.unknown`` governance signal -- it does not fail
        # them and does not silently drop them (see huginn/cli/contract_audit.py).
        for t in ("team.run.start", "campaign.control_trace", "event_bus.dropped"):
            assert t not in KNOWN_EVENT_TYPES
        report = lint({"type": "team.run.start", "timestamp": time.time(),
                       "source": "team"})
        assert "type.unknown" in _rules(report)
        assert report.ok  # advisory does not fail


class TestAgentEvent:
    def test_good_event_ok(self):
        report = lint({
            "type": "tool.result", "timestamp": time.time(),
            "source": "tool_adapter", "data": {"tool_name": "vasp"},
        })
        assert report.ok and report.violations == []

    def test_bad_type_naming_hard(self):
        report = lint({"type": "ToolResult", "timestamp": time.time(), "source": "x"})
        assert "type.naming" in _rules(report)
        assert not report.ok

    def test_unknown_type_advisory(self):
        report = lint({"type": "custom.thing", "timestamp": time.time(), "source": "x"})
        assert "type.unknown" in _rules(report)
        assert report.ok  # advisory does not fail

    def test_missing_timestamp_hard(self):
        report = lint({"type": "tool.result", "source": "x"})
        assert "envelope.timestamp" in _rules(report)

    def test_missing_source_advisory(self):
        report = lint({"type": "tool.result", "timestamp": time.time()})
        assert "envelope.source" in _rules(report)
        assert report.ok

    def test_payload_contract_missing_field_hard(self):
        # decision.point declares id/kind/status/narrative
        report = lint({"type": "decision.point", "timestamp": time.time(),
                       "source": "arbiter", "data": {"id": "d1"}})
        assert "payload.required" in _rules(report)
        assert not report.ok

    def test_alias_drift_advisory(self):
        report = lint({"type": "tool.result", "timestamp": time.time(),
                       "source": "tool_adapter", "data": {"tool": "vasp"}})
        assert "field.alias" in _rules(report)
        assert report.ok

    def test_non_serializable_data_advisory(self):
        report = lint({"type": "tool.result", "timestamp": time.time(),
                       "source": "x", "data": {"obj": object()}})
        assert "payload.json" in _rules(report)

    def test_audit_accepts_dataclass(self):
        ev = AgentEvent(type="tool.result", timestamp=time.time(),
                        source="tool_adapter", data={"tool_name": "vasp"})
        assert audit_agent_event(ev) == []


class TestSubagentResult:
    def test_success_with_error_is_contradiction(self):
        report = lint({"summary": "done", "success": True, "error": "boom"})
        assert "result.contradiction" in _rules(report)
        assert not report.ok

    def test_success_with_empty_summary_advisory(self):
        report = lint({"summary": "", "success": True, "full_output": "x"})
        assert "result.empty" in _rules(report)

    def test_both_empty_hard(self):
        report = lint({"summary": "", "success": False, "error": "e", "full_output": ""})
        hard = [v for v in report.violations if v.rule == "result.empty"]
        assert any(v.severity == "hard" for v in hard)

    def test_unexplained_failure_advisory(self):
        report = lint({"summary": "partial", "success": False, "full_output": "x"})
        assert "result.unexplained" in _rules(report)

    def test_clean_result_ok(self):
        report = lint({"summary": "found 3 phases", "success": True, "full_output": "..."})
        assert report.ok and report.violations == []

    def test_audit_accepts_object(self):
        class R:
            summary = "ok"
            full_output = "ok"
            success = True
            error = None

        assert audit_subagent_result(R()) == []


class TestAgentMessage:
    def test_bad_role_hard(self):
        report = lint({"role": "robot", "content": "hi"})
        assert "message.role" in _rules(report)
        assert not report.ok

    def test_empty_content_advisory(self):
        report = lint({"role": "user", "content": "  "})
        assert "message.content" in _rules(report)

    def test_good_message_ok(self):
        report = lint({"role": "assistant", "content": "done"})
        assert report.ok and report.violations == []

    def test_audit_accepts_dataclass(self):
        assert audit_agent_message(AgentMessage(role="user", content="hi")) == []


class TestSurfaceInference:
    def test_infer_result(self):
        assert lint({"summary": "s", "success": True}).surface == "result"

    def test_infer_message(self):
        assert lint({"role": "user", "content": "x"}).surface == "message"

    def test_unknown_surface(self):
        report = lint({"totally": "unknown"})
        assert "surface.unknown" in _rules(report)

    def test_explicit_surface_overrides(self):
        report = lint({"type": "tool.result", "timestamp": time.time(),
                       "source": "x"}, surface="event")
        assert report.surface == "event"


class TestWorkflowEvent:
    def test_missing_workflow_name_hard(self):
        report = lint({"type": "wf", "workflow_name": "", "stage_name": "s"},
                      surface="workflow")
        assert "workflow.name" in _rules(report)
        assert not report.ok


class TestEventBusAudit:
    async def test_good_event_counts_nothing(self):
        bus = EventBus()
        await bus.publish(AgentEvent(type="tool.result", timestamp=time.time(),
                                     source="tool_adapter", data={"tool_name": "vasp"}))
        assert bus.contract_stats() == {}

    async def test_alias_event_counted(self):
        bus = EventBus()
        await bus.publish(AgentEvent(type="tool.result", timestamp=time.time(),
                                     source="tool_adapter", data={"tool": "vasp"}))
        assert bus.contract_stats().get("field.alias", 0) >= 1

    async def test_hard_violation_counted(self):
        bus = EventBus()
        await bus.publish(AgentEvent(type="decision.point", timestamp=time.time(),
                                     source="arbiter", data={}))
        assert bus.contract_stats().get("payload.required", 0) >= 1

    async def test_fail_open(self):
        bus = EventBus()
        # non-serializable data must not raise out of publish
        await bus.publish(AgentEvent(type="tool.result", timestamp=time.time(),
                                     source="x", data={"obj": object()}))
        assert bus.contract_stats().get("payload.json", 0) >= 1


class TestCommsTool:
    async def test_flags_violation(self):
        tool = CommsLintTool()
        result = await tool.call(
            CommsLintInput(payload={"summary": "s", "success": True, "error": "boom"}), _ctx())
        assert result.success is True
        assert result.data["hard_count"] >= 1
        assert "result.contradiction" in result.data["summary"]

    async def test_clean_payload(self):
        tool = CommsLintTool()
        result = await tool.call(
            CommsLintInput(payload={"type": "tool.result", "timestamp": time.time(),
                                    "source": "x", "data": {"tool_name": "t"}}), _ctx())
        assert result.success is True
        assert result.data["hard_count"] == 0
        assert "contract satisfied" in result.data["summary"]


class TestCommsLifecycle:
    async def test_mounts_both_tools(self):
        from huginn.plugins.asd_ste100.main import AsdSte100Star

        star = AsdSte100Star()
        await star.on_load()
        try:
            assert SEGMENT_NAME in registered_prompt_segments()
            assert ToolRegistry.get(TOOL_NAME) is not None
            assert ToolRegistry.get(COMMS_TOOL_NAME) is not None
        finally:
            await star.on_unload()
        assert ToolRegistry.get(TOOL_NAME) is None
        assert ToolRegistry.get(COMMS_TOOL_NAME) is None
