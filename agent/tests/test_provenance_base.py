"""Tests for automatic provenance capture in HuginnTool.

The base class wraps ``_execute`` with a snapshot that lands in a contextvar
collector. These cover: snapshot construction, the set/get collector helpers,
capture-on-call, the no-collector short-circuit, and the input/output fields
landing on the snapshot.
"""

from __future__ import annotations

import asyncio

import pytest

from huginn.provenance import ProvenanceSnapshot
from huginn.tools.base import (
    HuginnTool,
    get_provenance_collector,
    set_provenance_collector,
)
from huginn.types import ToolResult


# A minimal tool that opts into the new _execute override point, so it inherits
# the base-class call() wrapper (and thus automatic provenance capture).
class _EchoTool(HuginnTool):
    name = "echo_tool"
    version = "2.3"

    async def _execute(self, args, context):  # noqa: ANN001 - test stub
        return ToolResult(data={"echoed": args.get("x", 0)}, success=True)


# A legacy-style tool that overrides call() directly, shadowing the wrapper.
class _LegacyTool(HuginnTool):
    name = "legacy_tool"

    async def call(self, args, context):  # noqa: ANN001 - test stub
        # does its own thing, including no automatic snapshot
        return ToolResult(data={"legacy": True}, success=True)


@pytest.fixture(autouse=True)
def _reset_collector():
    """The provenance collector is a context var that would otherwise leak
    between tests. Reset it before and after each test for isolation."""
    set_provenance_collector(None)
    yield
    set_provenance_collector(None)


# ── ProvenanceSnapshot construction ────────────────────────────────────────


class TestProvenanceSnapshot:
    def test_creation_and_fields(self):
        snap = ProvenanceSnapshot(
            timestamp="2026-07-06T00:00:00+00:00",
            tool_name="t",
            tool_version="1.0",
            input_params={"a": 1},
            output_hash="abc123def456",
        )
        assert snap.tool_name == "t"
        assert snap.tool_version == "1.0"
        assert snap.input_params == {"a": 1}
        assert snap.output_hash == "abc123def456"

    def test_to_dict_roundtrip(self):
        snap = ProvenanceSnapshot(
            timestamp="ts",
            tool_name="t",
            tool_version="1.0",
            input_params={"encut": 520},
            output_hash="deadbeef",
        )
        d = snap.to_dict()
        assert d["tool_name"] == "t"
        assert d["input_params"] == {"encut": 520}
        assert d["output_hash"] == "deadbeef"


# ── collector helpers ───────────────────────────────────────────────────────


class TestCollectorHelpers:
    def test_default_is_none(self):
        assert get_provenance_collector() is None

    def test_set_and_get_roundtrip(self):
        col: list = []
        set_provenance_collector(col)
        assert get_provenance_collector() is col
        set_provenance_collector(None)
        assert get_provenance_collector() is None


# ── HuginnTool.call provenance capture ─────────────────────────────────────


def _run(coro):
    """asyncio.run copies the caller's context into the task, so a collector
    set before run() is visible inside the awaited call()."""
    return asyncio.run(coro)


class TestCallCapture:
    def test_creates_snapshot_when_collector_set(self):
        collector: list = []
        set_provenance_collector(collector)

        tool = _EchoTool()
        result = _run(tool.call({"x": 5}, None))

        assert result.success
        assert len(collector) == 1
        snap = collector[0]
        assert isinstance(snap, ProvenanceSnapshot)
        assert snap.tool_name == "echo_tool"
        assert snap.tool_version == "2.3"

    def test_no_snapshot_when_collector_none(self):
        set_provenance_collector(None)
        tool = _EchoTool()
        # _capture_provenance must short-circuit and return None
        assert tool._capture_provenance({"x": 1}, ToolResult(data={}, success=True)) is None
        assert get_provenance_collector() is None
        # and a real call must not raise just because there's nowhere to store
        result = _run(tool.call({"x": 1}, None))
        assert result.success

    def test_snapshot_captures_input_params_and_output_hash(self):
        collector: list = []
        set_provenance_collector(collector)
        tool = _EchoTool()
        _run(tool.call({"x": 7, "y": 3}, None))

        snap = collector[0]
        assert snap.input_params == {"x": 7, "y": 3}
        assert snap.output_hash
        assert len(snap.output_hash) == 16
        assert snap.timestamp  # non-empty

    def test_same_output_produces_stable_hash(self):
        collector_a: list = []
        collector_b: list = []
        tool = _EchoTool()

        set_provenance_collector(collector_a)
        _run(tool.call({"x": 1}, None))
        set_provenance_collector(collector_b)
        _run(tool.call({"x": 1}, None))

        assert collector_a[0].output_hash == collector_b[0].output_hash

    def test_legacy_tool_overriding_call_is_not_double_captured(self):
        # _LegacyTool overrides call() directly, shadowing the wrapper.
        # It must still work, and the wrapper must not inject a snapshot.
        collector: list = []
        set_provenance_collector(collector)
        tool = _LegacyTool()
        result = _run(tool.call({"x": 1}, None))

        assert result.success
        assert result.data == {"legacy": True}
        assert collector == []  # wrapper was shadowed -> no auto snapshot

    def test_no_snapshot_when_provenance_flag_disabled(self, monkeypatch):
        from huginn.feature_flags import FeatureFlags

        monkeypatch.setattr(
            FeatureFlags,
            "is_enabled",
            lambda self, feature: False if feature == "provenance" else True,
        )
        collector: list = []
        set_provenance_collector(collector)
        tool = _EchoTool()
        _run(tool.call({"x": 1}, None))

        assert collector == []  # flag off -> capture skipped
