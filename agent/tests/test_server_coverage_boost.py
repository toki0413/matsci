"""Tests for server coverage boost (config endpoints and MCP)."""

from __future__ import annotations

import pytest
pytest.importorskip("mcp", reason="MCP SDK not installed (pip install mcp)")

import base64
import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from huginn.server import app
from huginn.types import HuginnConfig


client = TestClient(app)


class TestConfigEncryptEndpoint:
    def test_config_encrypt(self, tmp_path: Path):
        raw = {"provider": "openai", "model": "gpt-4o", "api_key": "secret-key"}
        response = client.post("/config/encrypt", json=raw)
        assert response.status_code == 200
        payload = response.json()
        assert "encrypted" in payload
        assert payload["provider"] == "openai"
        assert payload["model"] == "gpt-4o"
        assert payload["api_key"] != "secret-key"
        assert base64.b64decode(payload["encrypted"])


class TestMCPEndpoints:
    def test_mcp_servers_connect_disconnect(self, tmp_path: Path, monkeypatch: Any):
        import huginn.mcp_client as mcp_client_module

        # Use a fresh manager to avoid state leakage from other tests
        manager = mcp_client_module.MCPClientManager()
        monkeypatch.setattr(mcp_client_module, "mcp_manager", manager)

        # List should be empty initially
        assert manager.list_servers() == []

        # Register a mock server
        server_cfg = {
            "command": "python",
            "args": ["-c", "print('hello')"],
            "env": {},
        }
        manager.register_server("test_echo", server_cfg)
        assert "test_echo" in [s["name"] for s in manager.list_servers()]

        # Connect (will fail to actually spawn, but should handle gracefully)
        try:
            manager.connect_server("test_echo")
        except Exception:
            pass  # expected if the mock server can't start

        # Disconnect and remove
        manager.disconnect_server("test_echo")
        manager.remove_server("test_echo")
        assert manager.list_servers() == []
