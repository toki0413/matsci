"""Tests for server internals (unit level).

Merged from:
  - test_server.py          : config/encrypt endpoint + MCP manager logic
  - test_server_context.py  : server-scoped ServerContext
  - test_server_endpoints.py: direct async endpoint functions in huginn.server

The HTTP routing layer (TestClient against the real app) lives separately in
test_server_fastapi.py.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import tempfile
from pathlib import Path
from typing import Any

import pytest

from huginn.server_context import (
    ServerContext,
    create_server_context,
    set_server_context,
)


@pytest.fixture(scope="module", autouse=True)
def _bind_test_client(app_client):
    """Bind module-global `client` to the shared, properly-closed app_client.

    Replaces the old module-level ``client = TestClient(app)`` which leaked an
    anyio portal thread and never shut down the app lifespan (OOM + hang).
    """
    global client
    client = app_client
    yield
    client = None


@pytest.fixture(autouse=True)
def _dev_mode_and_isolated_config(tmp_path: Path, monkeypatch):
    """config/encrypt 需要 admin key, 测试只验加密逻辑不验 auth → 开 dev mode 绕过.

    同时把配置文件指到 tmp_path, 避免写到真实工作目录污染环境.
    /config/encrypt 加密的是当前运行时配置 (from env), 不是请求体,
    所以这里把 env 设成已知状态, 让断言可重复.
    """
    monkeypatch.setenv("HUGINN_DEV_MODE", "1")
    monkeypatch.setenv("HUGINN_CONFIG_FILE", str(tmp_path / "huginn.toml"))
    monkeypatch.setenv("HUGINN_PROVIDER", "openai")
    monkeypatch.setenv("HUGINN_MODEL", "gpt-4o")
    monkeypatch.setenv("HUGINN_API_KEY", "secret-key")
    # encrypt 端点用 module-level _config_path_override, 直接清掉前一个测试的残留
    import huginn.config as config_module
    from huginn.routes import config as config_routes

    monkeypatch.setattr(config_routes, "_config_path_override", None)
    # 清掉配置缓存, 让 _load_runtime_config 重新从 env 读
    monkeypatch.setattr(config_module, "_config_cache", None, raising=False)
    yield
    monkeypatch.setattr(config_routes, "_config_path_override", None)


# ── server-scoped context (from test_server_context.py) ──────────────────────


class TestServerContext:
    def test_create_server_context(self):
        ctx = create_server_context()
        assert isinstance(ctx, ServerContext)
        assert ctx.agent_factory is not None
        assert ctx.memory_manager is not None
        assert ctx.orchestrator is not None

    def test_set_server_context(self):
        ctx = create_server_context()
        set_server_context(ctx)
        from huginn.server_context import get_server_context

        assert get_server_context() is ctx


# ── direct async endpoint functions (from test_server_endpoints.py) ──────────


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


# ── config/encrypt endpoint (from test_server.py) ────────────────────────────


class TestConfigEncryptEndpoint:
    def test_config_encrypt(self, tmp_path: Path):
        # /config/encrypt 用 password 把当前运行时配置加密落盘.
        # 请求体只带 password, provider/model/api_key 来自 env (见 fixture).
        response = client.post(
            "/config/encrypt", json={"password": "test-password-123"}
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["success"] is True
        assert "path" in payload

        # 加密配置由 EncryptedConfig 整块加密落盘 (非 field-level mask),
        # 文件是密文二进制, 不能直接 json.loads. 用 password 解密回读校验.
        enc_path = Path(payload["path"])
        assert enc_path.exists(), f"encrypted config not written: {enc_path}"
        raw_bytes = enc_path.read_bytes()
        # 密文不应包含明文 api_key
        assert b"secret-key" not in raw_bytes, "api_key leaked in plaintext"

        from huginn.crypto import CryptoVault, EncryptedConfig

        vault = CryptoVault(master_password="test-password-123")
        ec = EncryptedConfig(config_path=enc_path, vault=vault)
        decrypted = ec.load()
        assert decrypted["provider"] == "openai"
        assert decrypted["model"] == "gpt-4o"
        assert decrypted["api_key"] == "secret-key"
        # Cleanup
        if enc_path.exists() and enc_path.parent != tmp_path:
            enc_path.unlink()


# ── MCP manager logic (from test_server.py) ──────────────────────────────────


class TestMCPEndpoints:
    def test_mcp_servers_connect_disconnect(self, tmp_path: Path, monkeypatch: Any):
        pytest.importorskip("mcp", reason="MCP SDK not installed (pip install mcp)")
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
        with contextlib.suppress(Exception):
            manager.connect_server("test_echo")  # expected if the mock server can't start

        # Disconnect and remove
        manager.disconnect_server("test_echo")
        manager.remove_server("test_echo")
        assert manager.list_servers() == []


class TestModelCaps:
    """/models/caps 端点逻辑: 按活跃模型返回能力标记 (vision/tools/reasoning/streaming)."""

    @staticmethod
    def _run(agents, models, monkeypatch):
        from types import SimpleNamespace

        from huginn.routes import config as config_routes

        def loader():
            return SimpleNamespace(agents=agents, models=models)

        monkeypatch.setattr(config_routes, "_load_runtime_config", loader)
        return asyncio.run(config_routes.get_model_caps())

    def test_text_model_vision_false(self, monkeypatch):
        r = self._run(
            [type("A", (), {"id": "lead", "enabled": True, "model_alias": "deepseek"})()],
            [type("M", (), {"alias": "deepseek", "model": "deepseek-v4-flash", "enabled": True})()],
            monkeypatch,
        )
        assert r["model"] == "deepseek-v4-flash"
        assert r["vision"] is False
        assert r["tools"] is True

    def test_vision_model_vision_true(self, monkeypatch):
        r = self._run(
            [type("A", (), {"id": "lead", "enabled": True, "model_alias": "gpt4o"})()],
            [type("M", (), {"alias": "gpt4o", "model": "gpt-4o", "enabled": True})()],
            monkeypatch,
        )
        assert r["model"] == "gpt-4o"
        assert r["vision"] is True
        assert r["tools"] is True

    def test_no_model_returns_fail_closed(self, monkeypatch):
        r = self._run([], [], monkeypatch)
        assert r["model"] is None
        assert r["vision"] is False
