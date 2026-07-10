"""Tests for deep polish — silent exceptions, plugin loading, path validation."""

import asyncio
import inspect
from pathlib import Path
from unittest.mock import patch

import pytest


# ── No more silent except: pass in agent.py ────────────────────────


class TestNoSilentExceptions:
    """Verify agent.py no longer has except Exception: pass."""

    def test_no_silent_pass_in_agent(self):
        """No `except Exception: pass` should remain in agent package."""
        import re
        agent_dir = Path(__file__).resolve().parent.parent / "huginn" / "agent"
        pattern = r"except Exception:\s*\n\s*pass\s*$"
        total = 0
        for f in agent_dir.glob("*.py"):
            content = f.read_text(encoding="utf-8")
            total += len(re.findall(pattern, content, re.MULTILINE))
        assert total == 0, (
            f"Found {total} silent 'except Exception: pass' in agent/"
        )


# ── No more traceback.print_exc in routes ────────────────────────


class TestNoTracebackPrint:
    """Verify routes no longer use traceback.print_exc()."""

    def test_no_traceback_print_exc_in_routes(self):
        routes_dir = Path(__file__).resolve().parent.parent / "huginn" / "routes"
        remaining = []
        for f in routes_dir.glob("*.py"):
            content = f.read_text(encoding="utf-8")
            if "traceback.print_exc()" in content:
                remaining.append(f.name)
        assert remaining == [], f"Still has traceback.print_exc(): {remaining}"


# ── Credentials endpoints have admin auth ─────────────────────────


class TestCredentialsAdminAuth:
    """Verify all credential endpoints require admin key."""

    def test_import_from_config_has_admin(self):
        from huginn.routes.credentials import router

        for route in router.routes:
            if hasattr(route, "path") and "import-from-config" in str(route.path):
                deps = getattr(route, "dependencies", [])
                assert deps, "import-from-config missing admin dependency"

    def test_link_model_has_admin(self):
        from huginn.routes.credentials import router

        for route in router.routes:
            if hasattr(route, "path") and "link-model" in str(route.path):
                deps = getattr(route, "dependencies", [])
                assert deps, "link-model missing admin dependency"


# ── Execution endpoint validates working_dir ──────────────────────


class TestExecutionPathValidation:
    """Verify /execute validates working_dir."""

    def test_validate_working_dir_rejects_outside(self):
        from huginn.routes.execution import _validate_working_dir
        from fastapi import HTTPException

        with patch("huginn.routes.execution.get_context") as mock_ctx:
            mock_ctx.return_value.config.workspace = "/tmp/test-ws"
            with pytest.raises(HTTPException) as exc:
                _validate_working_dir("/etc")
            assert exc.value.status_code == 403

    def test_validate_working_dir_accepts_inside(self, tmp_path):
        from huginn.routes.execution import _validate_working_dir

        sub = tmp_path / "work"
        sub.mkdir()
        with patch("huginn.routes.execution.get_context") as mock_ctx:
            mock_ctx.return_value.config.workspace = str(tmp_path)
            result = _validate_working_dir(str(sub))
            assert str(tmp_path) in result


# ── Star plugin loading in lifespan ───────────────────────────────


class TestStarPluginLoading:
    """Verify _load_star_plugins exists and is callable."""

    def test_load_star_plugins_exists(self):
        from huginn.lifespan import _load_star_plugins
        assert callable(_load_star_plugins)

    @pytest.mark.asyncio
    async def test_load_star_plugins_does_not_crash(self):
        """Loading plugins should not crash even if no plugins exist."""
        from huginn.lifespan import _load_star_plugins
        # Should complete without error
        await _load_star_plugins()


# ── call_with_fallback integrated in summarizer ───────────────────


class TestSummarizerFallback:
    """Verify the summarizer uses call_with_fallback on overload."""

    def test_summarizer_imports_fallback(self):
        """The _make_summarizer source should reference call_with_fallback."""
        from huginn.agent import HuginnAgent

        source = inspect.getsource(HuginnAgent._make_summarizer)
        assert "call_with_fallback" in source or "FallbackTriggeredError" in source


# ── Logger properly used in routes ────────────────────────────────


class TestRoutesUseLogger:
    """Verify route files use logger.error instead of traceback.print_exc."""

    def test_routes_have_logger(self):
        routes_dir = Path(__file__).resolve().parent.parent / "huginn" / "routes"
        for f in routes_dir.glob("*.py"):
            content = f.read_text(encoding="utf-8")
            if "logger.error" in content:
                assert "logger = " in content or "logging.getLogger" in content, \
                    f"{f.name} uses logger.error but has no logger defined"
