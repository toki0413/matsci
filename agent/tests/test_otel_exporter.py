"""Tests for the minimal OTLP/HTTP trace exporter and its wiring."""

from __future__ import annotations

import json

import pytest

import huginn.otel as otel
from huginn.otel import OtlpExporter, get_default_exporter
from huginn.telemetry import TelemetryCollector, TelemetrySpan


@pytest.fixture(autouse=True)
def _clean_default_exporter():
    otel.reset_default_exporter_for_tests()
    yield
    otel.reset_default_exporter_for_tests()


def _build_span_tree() -> TelemetrySpan:
    root = TelemetrySpan(name="agent_turn", metadata={"mode": "chat"})
    tool = TelemetrySpan(name="tool_call", metadata={"tool": "bash_tool", "success": True})
    tool.finish()
    root.children.append(tool)
    root.finish()
    return root


class _RecordingUrl:
    def __init__(self) -> None:
        self.requests: list[tuple[dict, bytes]] = []

    def __call__(self, req, timeout=5.0):
        data = req.data if isinstance(req.data, bytes) else b""
        self.requests.append((dict(req.headers), data))
        import io

        return io.BytesIO(b"{}")


def test_encode_otlp_json_structure(monkeypatch):
    recorder = _RecordingUrl()
    monkeypatch.setattr("huginn.otel.urllib.request.urlopen", recorder)
    exp = OtlpExporter(endpoint="http://localhost:9/v1/traces", service_name="huginn")

    exp.emit(_build_span_tree())
    exp.flush()

    assert len(recorder.requests) == 1
    headers, payload = recorder.requests[0]
    lowered = {k.lower(): v for k, v in headers.items()}
    assert lowered.get("content-type", "").startswith("application/json")
    body = json.loads(payload.decode("utf-8"))
    rspans = body["resourceSpans"]
    assert rspans[0]["resource"]["attributes"][0]["key"] == "service.name"
    spans = rspans[0]["scopeSpans"][0]["spans"]
    # root + one child
    assert len(spans) == 2
    by_name = {s["name"]: s for s in spans}
    root, tool = by_name["agent_turn"], by_name["tool_call"]
    assert root["kind"] == 2  # INTERNAL
    assert tool["kind"] == 3  # CLIENT
    assert tool["parentSpanId"] == root["spanId"]
    assert root["parentSpanId"] == ""
    # metadata serialized to attributes + status success
    attrs = {a["key"]: a["value"] for a in tool["attributes"]}
    assert attrs["tool"]["stringValue"] == "bash_tool"
    assert tool["status"]["code"] == 0
    assert int(tool["endTimeUnixNano"]) >= int(tool["startTimeUnixNano"])


def test_error_status_when_failed(monkeypatch):
    recorder = _RecordingUrl()
    monkeypatch.setattr("huginn.otel.urllib.request.urlopen", recorder)
    exp = OtlpExporter(endpoint="http://localhost:9/", interval_seconds=1.0)

    root = TelemetrySpan(name="tool_call", metadata={"success": False})
    root.finish()
    exp.emit(root)
    exp.flush()

    body = json.loads(recorder.requests[0][1].decode("utf-8"))
    span = body["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    assert span["status"]["code"] == 2  # ERROR


def test_fail_open_on_network_error(monkeypatch):
    def _boom(req, timeout=5.0):
        raise OSError("connection refused")

    monkeypatch.setattr("huginn.otel.urllib.request.urlopen", _boom)
    exp = OtlpExporter(endpoint="http://127.0.0.1:1/", interval_seconds=1.0)
    exp.emit(_build_span_tree())
    # Must not raise
    exp.flush()


def test_default_exporter_is_none_when_unconfigured(monkeypatch):
    monkeypatch.delenv("HUGINN_OTEL_ENDPOINT", raising=False)
    assert get_default_exporter() is None
    # Unconfigured collector never exports / never spawns a thread.
    collector = TelemetryCollector()
    assert collector._exporter is None


def test_default_exporter_from_env(monkeypatch):
    monkeypatch.setenv("HUGINN_OTEL_ENDPOINT", "http://localhost:4318/v1/traces")
    monkeypatch.setenv(
        "HUGINN_OTEL_HEADERS", json.dumps({"Authorization": "Basic dGVzdDp0ZXN0"})
    )
    exp = get_default_exporter()
    assert exp is not None
    assert exp.headers.get("Authorization") == "Basic dGVzdDp0ZXN0"


def test_collector_emits_root_to_exporter():
    emitted: list[TelemetrySpan] = []

    class FakeExporter:
        def emit(self, root):
            emitted.append(root)

        def flush(self):
            pass

        def shutdown(self, block=False):
            pass

    collector = TelemetryCollector(exporter=FakeExporter())
    with collector.span("agent_turn"), collector.span("inner"):
        pass
    # Only the root (after inner closes) is handed off.
    assert len(emitted) == 1
    assert emitted[0].name == "agent_turn"
    assert len(emitted[0].children) == 1
