"""Tests for FastAPI endpoints added to server.py."""

from __future__ import annotations

import asyncio
import base64
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def personas_path(tmp_path):
    return tmp_path / "personas.json"


class TestPersonaEndpoints:
    async def _call(self, func, *args, **kwargs):
        return await func(*args, **kwargs)

    def test_list_personas(self, personas_path, monkeypatch):
        import huginn.personas as personas_module
        from huginn.server import list_personas

        monkeypatch.setattr(
            personas_module, "_default_personas_path", lambda _=None: personas_path
        )
        result = asyncio.run(self._call(list_personas))
        assert "default" in [p["name"] for p in result["personas"]]
        assert result["default"] == "default"

    def test_create_and_get_persona(self, personas_path, monkeypatch):
        import huginn.personas as personas_module
        from huginn.server import create_persona, get_persona

        monkeypatch.setattr(
            personas_module, "_default_personas_path", lambda _=None: personas_path
        )
        created = asyncio.run(
            self._call(
                create_persona,
                {
                    "name": "api_bot",
                    "system_prompt": "You are API bot.",
                    "begin_dialogs": [{"role": "user", "content": "Hi"}],
                },
            )
        )
        assert created["success"] is True

        result = asyncio.run(self._call(get_persona, "api_bot"))
        assert result["success"] is True
        assert result["system_prompt"] == "You are API bot."


class TestUnifiedEndpoints:
    def test_unified_solve_endpoint(self):
        from huginn.server import unified_solve_endpoint

        result = asyncio.run(
            unified_solve_endpoint(
                {"model": "heat_equation_fem", "method": "fem", "n": 6}
            )
        )
        assert result["success"] is True
        assert result["method"] == "fem"
        assert result["n_dof"] == 7
        assert result["residual"] < 1e-10

    def test_unified_plot_endpoint(self):
        from huginn.server import unified_plot_endpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "plot.png"
            result = asyncio.run(
                unified_plot_endpoint(
                    {
                        "model": "linear_elasticity_fem",
                        "method": "fem",
                        "n": 5,
                        "output_path": str(output_path),
                    }
                )
            )
            assert result["success"] is True
            assert result["plot_path"] == str(output_path)
            assert base64.b64decode(result["plot_base64"])[:8] == b"\x89PNG\r\n\x1a\n"


class TestNewAgentEndpoints:
    def test_telemetry_summary_endpoint(self):
        from huginn.server import telemetry_summary

        result = asyncio.run(telemetry_summary())
        assert "summary" in result
        assert "total_spans" in result["summary"]

    def test_telemetry_spans_endpoint(self):
        from huginn.server import telemetry_spans

        result = asyncio.run(telemetry_spans())
        assert "spans" in result

    def test_memory_maintenance_endpoint(self):
        from huginn.server import memory_maintenance

        result = asyncio.run(memory_maintenance({}))
        assert result.get("success") is True
        assert "summary" in result

    def test_get_thread_endpoint(self):
        from huginn.server import get_thread

        result = asyncio.run(get_thread("unknown", None))
        assert result["exists"] is False
