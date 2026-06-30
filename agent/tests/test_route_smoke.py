"""Smoke tests for all route modules + config guidance + security."""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

pytest.importorskip("mcp", reason="MCP SDK not installed (pip install mcp)")

from fastapi.testclient import TestClient

from huginn.server import app

client = TestClient(app)


# ── Route smoke tests ─────────────────────────────────────────────
# Quick GET/POST hits to verify each route module is wired up and doesn't crash.


class TestRouteSmoke:
    """Smoke-test every registered route module."""

    def test_health(self):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert "status" in data
        assert "version" in data
        assert "configured" in data

    def test_health_guidance(self):
        r = client.get("/health/guidance")
        assert r.status_code == 200
        data = r.json()
        assert "configured" in data
        assert "available_providers" in data
        assert "keyless_providers" in data
        assert "supported_providers" in data
        assert isinstance(data["supported_providers"], list)
        assert "ollama" in data["supported_providers"]

    def test_health_rust(self):
        r = client.get("/health/rust")
        assert r.status_code == 200
        assert "available" in r.json()

    def test_tools_list(self):
        r = client.get("/tools")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_mcp_servers_list(self):
        r = client.get("/mcp/servers")
        assert r.status_code == 200
        assert "servers" in r.json()

    def test_mcp_status(self):
        r = client.get("/mcp/status")
        assert r.status_code == 200
        assert "servers" in r.json()

    def test_mcp_discover(self):
        r = client.get("/mcp/servers/discover")
        assert r.status_code == 200
        assert "servers" in r.json()

    def test_workflows_list(self):
        r = client.get("/workflows")
        assert r.status_code == 200

    def test_skills_list(self):
        r = client.get("/skills")
        assert r.status_code == 200

    def test_threads_list(self):
        r = client.get("/threads")
        assert r.status_code == 200

    def test_checkpoint_not_found(self):
        r = client.get("/checkpoints/nonexistent-id")
        assert r.status_code == 200
        assert r.json().get("exists") is False or "error" in r.json()

    def test_models_list(self):
        r = client.get("/models")
        assert r.status_code == 200

    def test_memory_list(self):
        r = client.get("/memory")
        assert r.status_code == 200

    def test_personas_list(self):
        r = client.get("/personas")
        assert r.status_code == 200

    def test_pet_status(self):
        r = client.get("/pet/status")
        assert r.status_code == 200

    def test_project_context(self):
        r = client.get("/project-context")
        assert r.status_code == 200

    def test_telemetry_summary(self):
        r = client.get("/telemetry/summary")
        assert r.status_code == 200

    def test_telemetry_spans(self):
        r = client.get("/telemetry/spans")
        assert r.status_code == 200

    def test_compat_firewall(self):
        r = client.get("/firewall/status")
        assert r.status_code == 200

    def test_codebase(self):
        r = client.get("/codebase")
        assert r.status_code == 200

    def test_hpc_status(self):
        r = client.post("/hpc/status", json={})
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is False
        assert "job_id" in data.get("error", "")

    def test_team_profiles(self):
        r = client.get("/team/profiles")
        assert r.status_code == 200


# ── Config guidance tests ─────────────────────────────────────────


class TestConfigGuidance:
    def test_guidance_detects_env_keys(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
        r = client.get("/health/guidance")
        data = r.json()
        providers = [p["provider"] for p in data["available_providers"]]
        assert "openai" in providers

    def test_guidance_no_keys(self, monkeypatch):
        # Remove all known API keys
        for key in [
            "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY",
            "GOOGLE_API_KEY", "OPENROUTER_API_KEY", "NVIDIA_API_KEY",
            "SILICONFLOW_API_KEY", "MOONSHOT_API_KEY", "ZHIPU_API_KEY",
            "BAICHUAN_API_KEY", "DASHSCOPE_API_KEY", "QIANFAN_API_KEY",
            "DOUBAO_API_KEY", "HUNYUAN_API_KEY",
        ]:
            monkeypatch.delenv(key, raising=False)
        monkeypatch.delenv("HUGINN_PROVIDER", raising=False)
        monkeypatch.delenv("HUGINN_API_KEY", raising=False)

        r = client.get("/health/guidance")
        data = r.json()
        assert data["configured"] is False
        assert data["recommendation"]["action"] in ("set_provider", "manual_setup")

    def test_guidance_supported_providers(self):
        r = client.get("/health/guidance")
        data = r.json()
        expected = {"anthropic", "openai", "ollama", "deepseek", "google-genai"}
        assert expected.issubset(set(data["supported_providers"]))


# ── Security: config whitelist ────────────────────────────────────


class TestConfigWhitelist:
    def test_unknown_keys_rejected(self):
        r = client.post("/config", json={"evil_env": "PATH=/malicious"})
        data = r.json()
        assert data["success"] is False
        assert "Unknown config keys" in data["error"]

    def test_valid_key_accepted(self, monkeypatch, tmp_path):
        # Ensure no side effects
        monkeypatch.delenv("HUGINN_PET_NAME", raising=False)
        monkeypatch.setenv("HUGINN_CONFIG_FILE", str(tmp_path / "test_config.toml"))
        r = client.post("/config", json={"pet_name": "TestBot"})
        data = r.json()
        assert data["success"] is True


# ── Security: tool strict validation ─────────────────────────────


class TestToolStrictValidation:
    def test_tool_not_found(self):
        r = client.post("/tools/nonexistent_tool", json={})
        data = r.json()
        assert "error" in data
        assert "not found" in data["error"]


# ── Health status accuracy ────────────────────────────────────────


class TestHealthStatus:
    def test_unconfigured_status(self, monkeypatch):
        monkeypatch.delenv("HUGINN_PROVIDER", raising=False)
        monkeypatch.delenv("HUGINN_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        r = client.get("/health")
        data = r.json()
        assert data["status"] == "unconfigured"
        assert data["configured"] is False

    def test_ollama_configured_without_key(self, monkeypatch):
        monkeypatch.setenv("HUGINN_PROVIDER", "ollama")
        r = client.get("/health")
        data = r.json()
        assert data["configured"] is True
        assert data["status"] == "ok"
        assert data["provider"] == "ollama"
        monkeypatch.delenv("HUGINN_PROVIDER", raising=False)
