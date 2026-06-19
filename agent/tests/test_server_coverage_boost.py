"""FastAPI endpoint coverage boost tests.

Exercises additional server.py endpoints not covered by test_server_fastapi.py
using the same module-scoped TestClient fixture pattern. All network, LLM, HPC,
MCP subprocess, and CoderRunner side effects are monkey-patched away.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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
    return tmp_path_factory.mktemp("server-coverage")


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
    old_context = server_module._context
    server_module._context = ctx
    yield ctx
    server_module._context = old_context


@pytest.fixture(scope="module")
def client(test_context, admin_key, module_tmp):
    """Return a TestClient with admin auth and isolated context."""
    from fastapi.testclient import TestClient

    import huginn.server as server_module

    os.environ["HUGINN_PERSONA"] = "default"
    server_module._checkpoints.clear()
    server_module._threads.clear()

    # Speed up agent creation: pretend Ollama is unavailable so server falls
    # back to the no-LLM mock agent path immediately.
    orig_check = server_module._check_ollama_available
    server_module._check_ollama_available = lambda *args, **kwargs: False

    with TestClient(server_module.app) as c:
        yield c

    server_module._check_ollama_available = orig_check


# ── Codebase endpoints ───────────────────────────────────────────


class TestCodebaseEndpoints:
    def test_codebase_status(self, client, test_context):
        class FakeCodebase:
            def status(self):
                return {"available": True, "indexed": 42}

        test_context.codebase = FakeCodebase()
        response = client.get("/codebase")
        assert response.status_code == 200
        data = response.json()
        assert data["available"] is True
        assert data["indexed"] == 42

    def test_codebase_index(self, client, test_context):
        class FakeCodebase:
            def index_workspace(self):
                return {"indexed": 7, "removed": 1}

        test_context.codebase = FakeCodebase()
        response = client.post("/codebase/index")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["indexed"] == 7

    def test_codebase_search(self, client, test_context):
        class FakeCodebase:
            def search(self, query, top_k):
                return [{"path": "foo.py", "text": f"result for {query}"}]

        test_context.codebase = FakeCodebase()
        response = client.post("/codebase/search", json={"query": "bar", "top_k": 3})
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 1
        assert data["results"][0]["path"] == "foo.py"


# ── Knowledge Base endpoints ─────────────────────────────────────


class TestKnowledgeEndpoints:
    def test_knowledge_upload_list_query_delete(self, client, test_context):
        docs = {"doc-1": {"id": "doc-1", "title": "hello"}}

        class FakeKB:
            def add_document(self, name, content):
                return {"id": "doc-1", "name": name, "size": len(content)}

            def list_documents(self):
                return list(docs.values())

            def count(self):
                return len(docs)

            def query(self, text, top_k):
                return [{"text": f"chunk for {text}"}]

            def delete_document(self, doc_id):
                return docs.pop(doc_id, None) is not None

        test_context.kb = FakeKB()

        response = client.post(
            "/knowledge/upload",
            files={"file": ("test.txt", b"hello world", "text/plain")},
        )
        assert response.status_code == 200
        assert response.json()["success"] is True

        response = client.get("/knowledge")
        assert response.status_code == 200
        data = response.json()
        assert data["available"] is True
        assert data["count"] == 1

        response = client.post("/knowledge/query", json={"query": "hi", "top_k": 2})
        assert response.status_code == 200
        assert "chunks" in response.json()

        response = client.delete("/knowledge/doc-1")
        assert response.status_code == 200
        assert response.json()["success"] is True


# ── Memory maintenance endpoints ─────────────────────────────────


class TestMemoryMaintenanceEndpoints:
    def test_memory_patch_promote_prune_sync(self, client):
        response = client.post(
            "/memory",
            json={
                "content": "test memory",
                "category": "fact",
                "importance": 0.9,
                "tier": "mid",
            },
        )
        assert response.status_code == 200
        memory_id = response.json()["memory_id"]

        response = client.patch(
            f"/memory/{memory_id}",
            json={"content": "updated memory", "tier": "long"},
        )
        assert response.status_code == 200
        assert response.json()["success"] is True

        response = client.post(f"/memory/promote/{memory_id}", json={})
        assert response.status_code == 200
        assert "success" in response.json()

        response = client.post("/memory/prune", json={"threshold": 0.1, "older_than_days": 1})
        assert response.status_code == 200
        data = response.json()
        assert "expired" in data or "error" in data

        response = client.post("/memory/sync-md")
        assert response.status_code == 200
        assert "path" in response.json()


# ── Agent / model endpoints ──────────────────────────────────────


class TestAgentModelEndpoints:
    def test_list_models(self, client):
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data or "error" in data

    def test_list_agents(self, client):
        response = client.get("/agents")
        assert response.status_code == 200
        assert "agents" in response.json()

    def test_chat_with_agent_mock(self, client, monkeypatch):
        import huginn.server as server_module

        class FakeAgent:
            def invoke(self, message):
                return {"messages": [SimpleNamespace(content=f"echo: {message}")]}

        class FakeFactory:
            def create(self, agent_id, **kwargs):
                return FakeAgent()

        monkeypatch.setattr(server_module, "get_agent_factory", FakeFactory)
        response = client.post(
            "/agents/lead/chat",
            json={"message": "hello", "thread_id": "t1"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "error" not in data
        assert "echo: hello" in data.get("content", "")


# ── Orchestration / team / coder / swarm ─────────────────────────


class TestOrchestrationEndpoints:
    def test_orchestrate(self, client, monkeypatch):
        import huginn.agents.orchestrator as orch_module

        fake_task = SimpleNamespace(
            task_id="t1", agent_id="lead", status="done", prompt="p"
        )
        fake_result = SimpleNamespace(
            success=True,
            objective="test",
            plan=SimpleNamespace(tasks=[fake_task]),
            outputs={"t1": "done"},
            summary="ok",
            error=None,
        )

        class FakeOrchestrator:
            def __init__(self, factory=None, memory_manager=None, max_concurrent=3):
                pass

            async def run(self, objective, on_status=None):
                return fake_result

        monkeypatch.setattr(orch_module, "Orchestrator", FakeOrchestrator)
        response = client.post("/orchestrate", json={"objective": "test"})
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["summary"] == "ok"

    def test_team_run(self, client, monkeypatch):
        import huginn.server as server_module

        fake_task = SimpleNamespace(
            task_id="t1", agent_id="lead", status="done", prompt="p"
        )
        fake_result = SimpleNamespace(
            success=True,
            objective="team test",
            plan=SimpleNamespace(tasks=[fake_task]),
            outputs={"t1": "out"},
            summary="team ok",
            error=None,
        )

        class FakeOrchestrator:
            async def run(self, objective, on_status=None):
                return fake_result

            async def plan(self, objective):
                return fake_result.plan

        monkeypatch.setattr(server_module, "get_orchestrator", lambda: FakeOrchestrator())
        response = client.post("/team/run", json={"objective": "team test"})
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["summary"] == "team ok"

    def test_coder(self, client, monkeypatch):
        import huginn.coder.loop as coder_module

        class FakeCoderRunner:
            def __init__(self, *args, **kwargs):
                pass

            def run(self, task, max_iterations=None):
                return {"final_answer": f"coded: {task}"}

        monkeypatch.setattr(coder_module, "CoderRunner", FakeCoderRunner)
        response = client.post("/coder", json={"task": "write hello", "auto_approve": True})
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "coded: write hello" in data["final_answer"]

    def test_swarm_run(self, client, monkeypatch):
        import huginn.agents.swarm as swarm_module

        class FakeSwarm:
            async def run(self, task):
                return {"task": task, "result": "swarm done"}

        monkeypatch.setattr(swarm_module, "HuginnSwarm", lambda workers: FakeSwarm())
        response = client.post("/swarm/run", json={"task": "explore"})
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["result"] == "swarm done"


# ── Benchmark / evolution / execute ──────────────────────────────


class TestBenchmarkEvolutionExecuteEndpoints:
    def test_bench_run(self, client, monkeypatch):
        import huginn.bench.runner as bench_module

        fake_report = SimpleNamespace(
            run_id="r1",
            total=1,
            passed=1,
            failed=0,
            skipped=0,
            metrics={"pass_rate": 1.0},
            results=[],
            evolution_report=None,
        )

        class FakeBenchmarkRunner:
            def __init__(self, *args, **kwargs):
                pass

            def run(self, evolve=False, categories=None):
                return fake_report

        monkeypatch.setattr(bench_module, "BenchmarkRunner", FakeBenchmarkRunner)
        response = client.post("/bench/run", json={"categories": ["math"]})
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["report"]["run_id"] == "r1"

    def test_evolve_run(self, client, monkeypatch):
        import huginn.evolution.engine as evolve_module
        import huginn.evolution.logger as logger_module

        class FakeEvolutionEngine:
            def __init__(self, *args, **kwargs):
                pass

            def run_full_evolution_cycle(self):
                return {"new_rules": 1}

        class FakeExecutionLogger:
            def __init__(self, *args, **kwargs):
                pass

        monkeypatch.setattr(evolve_module, "EvolutionEngine", FakeEvolutionEngine)
        monkeypatch.setattr(logger_module, "ExecutionLogger", FakeExecutionLogger)
        response = client.post("/evolve/run", json={})
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["report"]["new_rules"] == 1

    def test_execute(self, client):
        response = client.post(
            "/execute",
            json={
                "name": "test_execute",
                "working_dir": ".",
                "stages": [
                    {
                        "id": "echo",
                        "tool": "bash_tool",
                        "action": "run",
                        "params": {"command": ["echo", "hi"]},
                    }
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "stages" in data


# ── Config encryption ────────────────────────────────────────────


class TestConfigEncryptEndpoint:
    def test_config_encrypt(self, client, admin_key, module_tmp, monkeypatch):
        import huginn.server as server_module

        config_path = module_tmp / "huginn.toml"
        config_path.write_text('[huginn]\nprovider = "ollama"\n', encoding="utf-8")

        # The endpoint changes directory to the file's parent; keep cwd stable.
        monkeypatch.chdir(module_tmp)
        response = client.post(
            "/config/encrypt",
            headers={"X-HUGINN-ADMIN-API-KEY": admin_key},
            json={"path": str(config_path), "password": "secret123"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["path"].endswith(".enc")


# ── Unified / workflows / skills ─────────────────────────────────


class TestUnifiedWorkflowSkillEndpoints:
    def test_unified_derive(self, client):
        response = client.post(
            "/unified/derive",
            json={"model": "harmonic_oscillator_md"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "success" in data

    def test_workflows_execute_symbolic_verify(self, client):
        response = client.post(
            "/workflows/execute",
            json={
                "template": "symbolic_verify",
                "args": {
                    "verify_type": "derivative",
                    "expression": "x**2",
                    "symbols": ["x"],
                    "variable": "x",
                },
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "stages" in data

    def test_skills_execute_builtin(self, client):
        response = client.post(
            "/skills/execute",
            json={
                "skill": "symbolic_verify",
                "args": {
                    "verify_type": "derivative",
                    "expression": "x**2",
                    "symbols": ["x"],
                    "variable": "x",
                },
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "skill" in data
        assert "steps" in data


# ── MCP management ───────────────────────────────────────────────


class TestMCPEndpoints:
    def test_mcp_servers_connect_disconnect(self, client, monkeypatch):
        import huginn.mcp_client as mcp_client_module
        import huginn.server as server_module
        import huginn.tools.mcp_adapter as mcp_adapter_module

        class FakeMCPManager:
            def __init__(self):
                self._sessions = {"dummy": object()}
                self._tools = []

            async def connect(self, config):
                return None

            async def disconnect(self, name):
                return None

        fake_tool = SimpleNamespace(name="dummy_tool", description="d", server_name="dummy")

        def fake_register(manager):
            return [fake_tool]

        monkeypatch.setattr(mcp_client_module, "MCPClientManager", FakeMCPManager)
        monkeypatch.setattr(mcp_adapter_module, "register_mcp_tools", fake_register)
        test_context = server_module.get_context()
        test_context.mcp_manager = None

        response = client.post(
            "/mcp/servers/connect",
            json={"name": "dummy", "command": "python", "args": ["server.py"]},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert any(t["name"] == "dummy_tool" for t in data["tools"])

        response = client.post("/mcp/servers/dummy/disconnect")
        assert response.status_code == 200
        assert response.json()["success"] is True


# ── Sandbox / analyze / viz ──────────────────────────────────────


class TestSandboxAnalyzeVizEndpoints:
    def test_sandbox_execute(self, client):
        response = client.post(
            "/sandbox/execute",
            json={"code": "print(1+1)", "timeout_seconds": 5},
        )
        assert response.status_code == 200
        data = response.json()
        assert "success" in data

    def test_analyze_spectral(self, client):
        response = client.post("/analyze/spectral", json={})
        assert response.status_code == 200
        assert "error" in response.json()

    def test_analyze_dynamics(self, client):
        response = client.post("/analyze/dynamics", json={})
        assert response.status_code == 200
        assert "error" in response.json()

    def test_analyze_tda(self, client):
        response = client.post("/analyze/tda", json={})
        assert response.status_code == 200
        assert "error" in response.json()

    def test_analyze_sindy(self, client):
        response = client.post("/analyze/sindy", json={})
        assert response.status_code == 200
        assert "error" in response.json()

    def test_viz_persistence(self, client):
        response = client.post("/viz/persistence", json={})
        assert response.status_code == 200
        data = response.json()
        assert data["fallback"] is True
        assert "html" in data

    def test_viz_sindy(self, client):
        response = client.post("/viz/sindy", json={})
        assert response.status_code == 200
        data = response.json()
        assert data["fallback"] is True
        assert "html" in data


# ── WebSocket ────────────────────────────────────────────────────


class TestWebSocket:
    def test_websocket_ping_pong(self, client):
        with client.websocket_connect("/ws/agent") as ws:
            ws.send_json({"type": "ping"})
            data = ws.receive_json()
            assert data["type"] == "pong"

    def test_websocket_user_input_mock_mode(self, client):
        with client.websocket_connect("/ws/agent") as ws:
            ws.send_json(
                {
                    "type": "user_input",
                    "content": "hello",
                    "thread_id": "ws-test",
                }
            )
            data = ws.receive_json()
            assert "type" in data
