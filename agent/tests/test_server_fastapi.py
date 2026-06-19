"""FastAPI integration tests using TestClient.

These tests exercise the real HTTP routing layer and cover endpoints that are
not reached by the direct async function tests in test_server_endpoints.py.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from huginn.config import HuginnConfig
from huginn.server_context import ServerContext, create_server_context, set_server_context


async def _noop():
    pass


@pytest.fixture(scope="module", autouse=True)
def _patch_mcp_lifespan_module():
    """Disable MCP subprocess initialization in tests to keep startup fast."""
    import huginn.server as server_module

    original_init = server_module._init_mcp_tools
    original_shutdown = server_module._shutdown_mcp
    server_module._init_mcp_tools = _noop
    server_module._shutdown_mcp = _noop
    yield
    server_module._init_mcp_tools = original_init
    server_module._shutdown_mcp = original_shutdown


@pytest.fixture(scope="module")
def admin_key():
    key = "test-admin-key"
    os.environ["HUGINN_ADMIN_API_KEY"] = key
    return key


@pytest.fixture(scope="module")
def module_tmp(tmp_path_factory):
    return tmp_path_factory.mktemp("server-fastapi")


@pytest.fixture(scope="module")
def test_context(module_tmp):
    """Provide an isolated server context and reset the module global."""
    import huginn.server as server_module

    os.environ["HUGINN_PROVIDER"] = "ollama"
    os.environ["HUGINN_MODEL"] = "qwen2.5:14b"
    os.environ["HUGINN_WORKSPACE"] = str(module_tmp)

    cfg = HuginnConfig(provider="ollama", model="qwen2.5:14b", workspace=str(module_tmp))
    ctx = create_server_context(cfg)
    set_server_context(ctx)
    # Reset the module-level cache so server.py picks up the new context.
    old_context = server_module._context
    server_module._context = ctx
    yield ctx
    server_module._context = old_context


@pytest.fixture(scope="module")
def client(test_context, admin_key, module_tmp):
    """Return a TestClient with admin auth and isolated context."""
    from fastapi.testclient import TestClient

    import huginn.server as server_module

    # Point personas at the temp workspace and clear any cached app state.
    os.environ["HUGINN_PERSONA"] = "default"
    server_module._checkpoints.clear()
    server_module._threads.clear()

    with TestClient(server_module.app) as c:
        yield c


class TestHealthEndpoints:
    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_health_rust(self, client):
        response = client.get("/health/rust")
        assert response.status_code == 200
        data = response.json()
        assert data["available"] is False
        assert "huginn_ext" in data["module"]


class TestConfigEndpoints:
    def test_get_config_requires_admin(self, client):
        response = client.get("/config")
        assert response.status_code == 401

    def test_get_config(self, client, admin_key):
        response = client.get("/config", headers={"X-HUGINN-ADMIN-API-KEY": admin_key})
        assert response.status_code == 200
        data = response.json()
        assert data["provider"] == "ollama"
        assert data["api_key"] in ("***", None)

    def test_update_config(self, client, admin_key, monkeypatch):
        response = client.post(
            "/config",
            headers={"X-HUGINN-ADMIN-API-KEY": admin_key},
            json={"persona": "dft_expert", "pet_name": "TestRaven"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["config"]["persona"] == "dft_expert"
        assert data["config"]["pet_name"] == "TestRaven"


class TestProjectContextEndpoints:
    def test_get_project_context(self, client):
        response = client.get("/project-context")
        assert response.status_code == 200
        data = response.json()
        assert "content" in data
        assert "path" in data

    def test_update_project_context(self, client):
        response = client.post("/project-context", json={"content": "# Notes"})
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


class TestPersonaEndpointsFastAPI:
    def test_list_personas(self, client):
        response = client.get("/personas")
        assert response.status_code == 200
        data = response.json()
        assert "default" in [p["name"] for p in data["personas"]]

    def test_get_persona(self, client):
        response = client.get("/personas/default")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["name"] == "default"

    def test_create_persona(self, client):
        response = client.post(
            "/personas",
            json={
                "name": "api_test",
                "system_prompt": "You are a test persona.",
                "description": "Test persona",
                "when_to_use": ["testing"],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["persona"]["name"] == "api_test"

    def test_match_persona(self, client):
        client.post(
            "/personas",
            json={
                "name": "dft_expert",
                "system_prompt": "DFT expert",
                "description": "Density functional theory",
                "when_to_use": ["DFT", "VASP"],
            },
        )
        response = client.post("/personas/match", json={"query": "Run VASP DFT"})
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert any(m["name"] == "dft_expert" for m in data["matches"])

    def test_set_default_persona(self, client):
        client.post(
            "/personas",
            json={"name": "reviewer", "system_prompt": "Reviewer"},
        )
        response = client.patch("/personas/reviewer/default")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["default"] == "reviewer"

    def test_switch_persona(self, client):
        response = client.post("/personas/default/switch", json={})
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["persona"] == "default"

    def test_delete_persona(self, client):
        client.post(
            "/personas",
            json={"name": "to_delete", "system_prompt": "temp"},
        )
        response = client.delete("/personas/to_delete")
        assert response.status_code == 200
        assert response.json()["success"] is True

    def test_get_persona_emotion(self, client):
        response = client.get("/personas/default/emotion")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "state" in data


class TestAgentEndpointsFastAPI:
    def test_list_agents(self, client):
        response = client.get("/agents")
        assert response.status_code == 200
        data = response.json()
        assert "agents" in data

    def test_chat_with_agent_mock_mode(self, client, test_context):
        # Force a mock agent by clearing the factory profiles.
        test_context.agent_factory._profiles = {}
        response = client.post(
            "/agents/lead/chat",
            json={"message": "hello", "thread_id": "t1"},
        )
        assert response.status_code == 200
        data = response.json()
        # Mock mode returns an error because there is no LLM.
        assert "error" in data or "content" in data


class TestExploreEndpointFastAPI:
    def test_explore_http(self, client):
        response = client.post(
            "/explore",
            json={
                "objective": "minimize energy",
                "max_iterations": 2,
                "max_branches": 3,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "success" in data


class TestThreadEndpointsFastAPI:
    def test_list_threads(self, client):
        response = client.get("/threads")
        assert response.status_code == 200
        data = response.json()
        assert "threads" in data

    def test_get_thread(self, client):
        response = client.get("/threads/unknown")
        assert response.status_code == 200
        data = response.json()
        assert data["exists"] is False


class TestTelemetryEndpointsFastAPI:
    def test_telemetry_summary(self, client):
        response = client.get("/telemetry/summary")
        assert response.status_code == 200
        data = response.json()
        assert "summary" in data

    def test_telemetry_spans(self, client):
        response = client.get("/telemetry/spans")
        assert response.status_code == 200
        data = response.json()
        assert "spans" in data


class TestMemoryMaintenanceEndpointFastAPI:
    def test_memory_maintenance(self, client):
        response = client.post("/memory/maintenance", json={})
        assert response.status_code == 200
        data = response.json()
        assert data.get("success") is True
        assert "summary" in data


class TestMathAnythingCompatibility:
    def test_firewall_status(self, client):
        response = client.get("/firewall/status")
        assert response.status_code == 200
        assert response.json()["enabled"] is False

    def test_analyze_symmetry(self, client):
        response = client.post("/analyze/symmetry", json={})
        assert response.status_code == 200
        assert "error" in response.json()


class TestToolsEndpoints:
    def test_list_tools(self, client):
        response = client.get("/tools")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert any(t.get("function", {}).get("name") == "bash_tool" for t in data)

    def test_call_tool_not_found(self, client):
        response = client.post("/tools/nonexistent", json={})
        assert response.status_code == 200
        assert "error" in response.json()

    def test_call_tool_invalid_input(self, client):
        response = client.post("/tools/bash_tool", json={})
        assert response.status_code == 200
        assert "error" in response.json()


class TestMemoryEndpoints:
    def test_create_and_list_memory(self, client):
        response = client.post(
            "/memory",
            json={
                "content": "silicon band gap is 1.12 eV",
                "category": "fact",
                "importance": 0.8,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get("success") is True
        memory_id = data["memory_id"]

        response = client.get("/memory", params={"category": "fact"})
        assert response.status_code == 200
        entries = response.json()["entries"]
        assert any(e.get("id") == memory_id for e in entries)

    def test_search_memory(self, client):
        response = client.post("/memory/search", json={"query": "silicon"})
        assert response.status_code == 200
        assert "results" in response.json()

    def test_memory_stats(self, client):
        response = client.get("/memory/stats")
        assert response.status_code == 200
        data = response.json()
        assert "longterm_entries" in data or "stats" in data


class TestThreadEndpointsExtended:
    def test_create_rename_delete_thread(self, client):
        response = client.post("/threads", json={"label": "test-thread"})
        assert response.status_code == 200
        data = response.json()
        thread_id = data["id"]
        assert data["label"] == "test-thread"

        response = client.patch(f"/threads/{thread_id}", json={"label": "renamed"})
        assert response.status_code == 200
        assert response.json()["label"] == "renamed"

        response = client.delete(f"/threads/{thread_id}")
        assert response.status_code == 200
        assert response.json()["success"] is True


class TestPetEndpoints:
    def test_pet_status(self, client):
        response = client.get("/pet/status")
        assert response.status_code == 200
        assert "level" in response.json()

    def test_pet_feed(self, client):
        response = client.post("/pet/feed", json={"treat": "berry"})
        assert response.status_code == 200
        data = response.json()
        assert "ok" in data and data["ok"] is True

    def test_pet_reset(self, client):
        response = client.post("/pet/reset")
        assert response.status_code == 200
        assert response.json()["ok"] is True


class TestModelEndpoints:
    def test_list_models(self, client):
        response = client.get("/models")
        assert response.status_code == 200
        assert "models" in response.json()


class TestExportEndpoint:
    def test_export_remote_jobs(self, client):
        response = client.get("/export", params={"source": "remote_jobs", "fmt": "json"})
        assert response.status_code == 200
        assert "huginn_export" in response.headers.get("content-disposition", "")


class TestDiagnoseEndpoint:
    def test_diagnose_vasp_edddav(self, client):
        response = client.post(
            "/diagnose",
            json={"error_message": "EDDDAV error", "software": "vasp"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get("success") is True


class TestUnifiedEndpoints:
    def test_unified_models(self, client):
        response = client.get("/unified/models")
        assert response.status_code == 200
        assert "models" in response.json()


class TestWorkflowEndpoints:
    def test_list_workflows(self, client):
        response = client.get("/workflows")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_execute_unknown_workflow(self, client):
        response = client.post("/workflows/execute", json={"template": "nonexistent"})
        assert response.status_code == 200
        assert "error" in response.json()


class TestSkillsEndpoints:
    def test_list_skills(self, client):
        response = client.get("/skills")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_execute_unknown_skill(self, client):
        response = client.post("/skills/execute", json={"skill": "nonexistent"})
        assert response.status_code == 200
        assert "error" in response.json()


class TestMCPEndpoints:
    def test_list_mcp_servers_no_manager(self, client):
        response = client.get("/mcp/servers")
        assert response.status_code == 200
        data = response.json()
        assert data["servers"] == []

    def test_mcp_servers_discover_no_manager(self, client):
        response = client.get("/mcp/servers/discover")
        assert response.status_code == 200
        assert "servers" in response.json()


class TestHPCEndpoints:
    def test_hpc_test_missing_credentials(self, client):
        response = client.post("/hpc/test", json={"host": "", "username": ""})
        assert response.status_code == 200
        assert response.json()["success"] is False

    def test_hpc_submit_missing_credentials(self, client):
        response = client.post("/hpc/submit", json={"host": "", "username": ""})
        assert response.status_code == 200
        assert response.json()["success"] is False

    def test_hpc_status_missing_job_id(self, client):
        response = client.post(
            "/hpc/status", json={"host": "h", "username": "u", "job_id": ""}
        )
        assert response.status_code == 200
        assert response.json()["success"] is False


class TestTeamEndpoints:
    def test_team_profiles(self, client):
        response = client.get("/team/profiles")
        assert response.status_code == 200
        assert "profiles" in response.json()

    def test_team_plan(self, client):
        response = client.post("/team/plan", json={"objective": "test plan"})
        assert response.status_code == 200
        data = response.json()
        assert "tasks" in data or "error" in data


class TestCheckpointEndpoints:
    def test_checkpoint_lifecycle(self, client, module_tmp):
        (module_tmp / "file.txt").write_text("hello", encoding="utf-8")
        response = client.post("/checkpoints", json={"path": str(module_tmp)})
        assert response.status_code == 200
        cp_id = response.json()["id"]

        response = client.get(f"/checkpoints/{cp_id}")
        assert response.status_code == 200
        assert "file.txt" in response.json()["files"]

        response = client.get(f"/checkpoints/{cp_id}/diff")
        assert response.status_code == 200
        assert "diffs" in response.json()

        response = client.post(f"/checkpoints/{cp_id}/reject")
        assert response.status_code == 200
        assert response.json()["success"] is True


class TestVizEndpoints:
    def test_viz_dos(self, client):
        response = client.post("/viz/dos", json={})
        assert response.status_code == 200
        assert "html" in response.json()

    def test_viz_phase(self, client):
        response = client.post("/viz/phase", json={})
        assert response.status_code == 200
        assert "html" in response.json()
