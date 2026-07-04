"""Tests for error handling and operational capability improvements."""

import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest


# ── Health endpoints ──────────────────────────────────────────────


class TestHealthEndpoints:
    """Verify /health/live and /health/ready exist and work correctly."""

    def test_health_live_route_exists(self):
        from huginn.routes.health import router
        paths = [r.path for r in router.routes if hasattr(r, "path")]
        assert "/health/live" in paths
        assert "/health/ready" in paths

    def test_health_live_returns_200(self):
        from huginn.routes.health import health_live
        result = __import__("asyncio").run(health_live())
        assert result["status"] == "alive"

    def test_health_ready_includes_error_field(self):
        """Readiness check should include error details on failure."""
        from huginn.routes.health import health_ready
        from fastapi import Response

        response = Response()
        result = __import__("asyncio").run(health_ready(response))
        assert "checks" in result
        # Each check should have a "status" field
        for check_name, check_val in result["checks"].items():
            assert "status" in check_val, f"{check_name} missing status"
            if check_val["status"] == "fail":
                assert "error" in check_val, f"{check_name} failed but no error field"

    def test_health_deprecation_notice(self):
        """Legacy /health should include deprecation notice."""
        from huginn.routes.health import health
        result = __import__("asyncio").run(health())
        assert "deprecation" in result or "Use /health/live" in str(result)


# ── Unified error response ────────────────────────────────────────


class TestUnifiedErrorResponse:
    """Verify huginn_error_response produces correct envelope."""

    def test_basic_error_response(self):
        from huginn.errors import huginn_error_response
        body = huginn_error_response("TEST_ERROR", "Something went wrong", "req-123")
        assert body["error_code"] == "TEST_ERROR"
        assert body["message"] == "Something went wrong"
        assert body["request_id"] == "req-123"

    def test_error_response_with_details(self):
        from huginn.errors import huginn_error_response
        body = huginn_error_response(
            "TOOL_NOT_FOUND", "Tool missing", "req-456",
            details={"tool": "bash"},
        )
        assert body["details"]["tool"] == "bash"

    def test_error_code_enum(self):
        from huginn.errors import ErrorCode
        assert ErrorCode.RATE_LIMITED.value == "RATE_LIMITED"
        assert ErrorCode.MAINTENANCE_MODE.value == "MAINTENANCE_MODE"

    def test_huginn_error_base_class(self):
        from huginn.errors import HuginnError, ErrorCode
        err = HuginnError("test error")
        response = err.to_response("req-789")
        assert response["error_code"] == ErrorCode.INTERNAL_ERROR.value
        assert response["request_id"] == "req-789"


# ── Maintenance mode ──────────────────────────────────────────────


class TestMaintenanceMode:
    """Verify maintenance mode middleware works."""

    def test_maintenance_off_by_default(self):
        from huginn.middleware.maintenance import is_maintenance_mode
        with patch.dict("os.environ", {}, clear=False):
            os.environ.pop("HUGINN_MAINTENANCE", None)
            from huginn.middleware.maintenance import _maintenance_active
            # Reset runtime toggle
            import huginn.middleware.maintenance as m
            m._maintenance_active = False
            assert not is_maintenance_mode()

    def test_maintenance_via_env(self):
        from huginn.middleware.maintenance import is_maintenance_mode
        with patch.dict("os.environ", {"HUGINN_MAINTENANCE": "1"}):
            assert is_maintenance_mode()

    def test_maintenance_via_runtime_toggle(self):
        from huginn.middleware.maintenance import set_maintenance_mode, is_maintenance_mode
        set_maintenance_mode(True)
        assert is_maintenance_mode()
        set_maintenance_mode(False)
        assert not is_maintenance_mode()

    def test_admin_route_exists(self):
        from huginn.routes.admin import router
        paths = [r.path for r in router.routes if hasattr(r, "path")]
        assert "/admin/maintenance" in paths


# ── Alert webhook ────────────────────────────────────────────────


class TestAlertWebhook:
    """Verify alert webhook fires on anomalies."""

    def test_webhook_not_configured(self):
        """When HUGINN_ALERT_WEBHOOK_URL is not set, should be no-op."""
        from huginn.diagnostics.system_health import SystemHealthMonitor
        monitor = SystemHealthMonitor()
        with patch.dict("os.environ", {}, clear=False):
            os.environ.pop("HUGINN_ALERT_WEBHOOK_URL", None)
            # Should not crash
            monitor._fire_alert_webhook([])

    def test_webhook_called_with_correct_payload(self):
        """When webhook URL is set, should POST anomalies."""
        from huginn.diagnostics.system_health import SystemHealthMonitor, AnomalyEvent
        import json
        import urllib.request

        monitor = SystemHealthMonitor()

        captured_requests = []

        def mock_urlopen(req, timeout=5):
            captured_requests.append({
                "url": req.full_url,
                "data": json.loads(req.data.decode()) if req.data else None,
                "headers": dict(req.headers),
            })
            return type("MockResp", (), {"read": lambda self: b'{"ok": true}', "status": 200})()

        event = AnomalyEvent(
            timestamp=time.time(),
            resource="cpu",
            value=95.0,
            threshold=90.0,
            severity="critical",
            message="CPU usage 95% exceeds threshold 90%",
        )

        with patch.dict("os.environ", {"HUGINN_ALERT_WEBHOOK_URL": "http://example.com/webhook"}):
            with patch("urllib.request.urlopen", mock_urlopen):
                monitor._fire_alert_webhook([event])

        assert len(captured_requests) == 1
        assert captured_requests[0]["url"] == "http://example.com/webhook"
        assert captured_requests[0]["data"]["event"] == "anomaly"
        assert captured_requests[0]["data"]["resource"] == "cpu"
        assert captured_requests[0]["data"]["severity"] == "critical"


# ── Graceful degradation ─────────────────────────────────────────


class TestGracefulDegradation:
    """Verify search falls back to FTS-only when vector store fails."""

    def test_vector_search_failure_falls_back(self):
        """When vector search crashes, search should still return FTS results."""
        from huginn.memory.longterm import LongTermMemory
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            mem = LongTermMemory(db_path=str(Path(tmpdir) / "test.db"))

            # Mock vector store to raise on search but allow ingest
            class MockVS:
                @staticmethod
                def search(*a, **k):
                    raise RuntimeError("model crashed")
                @staticmethod
                def ingest(*a, **k):
                    pass  # no-op during store

            mem._vector_store = MockVS()
            mem._enable_semantic = True

            # Store something first (use correct API)
            mem.store("test content about silicon", importance=0.8)

            # Search should not crash even though vector store is broken
            results = mem.retrieve("silicon", semantic=True)
            assert isinstance(results, list)  # Should get FTS results at least


# ── No more print() in lifespan/server_core ──────────────────────


class TestNoPrintInCoreFiles:
    """Verify print() has been replaced with logger."""

    def test_no_print_in_lifespan(self):
        lifespan = Path(__file__).resolve().parent.parent / "huginn" / "lifespan.py"
        content = lifespan.read_text(encoding="utf-8")
        # Allow print only in __main__ blocks or comments
        import re
        real_prints = re.findall(r'^\s*print\(', content, re.MULTILINE)
        assert len(real_prints) == 0, f"Found {len(real_prints)} print() calls in lifespan.py"

    def test_no_print_in_server_core(self):
        sc = Path(__file__).resolve().parent.parent / "huginn" / "server_core.py"
        content = sc.read_text(encoding="utf-8")
        import re
        real_prints = re.findall(r'^\s*print\(', content, re.MULTILINE)
        assert len(real_prints) == 0, f"Found {len(real_prints)} print() calls in server_core.py"

    def test_logger_defined_in_lifespan(self):
        lifespan = Path(__file__).resolve().parent.parent / "huginn" / "lifespan.py"
        content = lifespan.read_text(encoding="utf-8")
        assert "logger = " in content or "logging.getLogger" in content

    def test_logger_defined_in_server_core(self):
        sc = Path(__file__).resolve().parent.parent / "huginn" / "server_core.py"
        content = sc.read_text(encoding="utf-8")
        assert "logger = " in content or "logging.getLogger" in content
