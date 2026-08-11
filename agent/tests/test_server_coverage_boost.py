"""Tests for server coverage boost (config endpoints and MCP)."""

from __future__ import annotations

import pytest

pytest.importorskip("mcp", reason="MCP SDK not installed (pip install mcp)")

import contextlib
from pathlib import Path
from typing import Any


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
        with contextlib.suppress(Exception):
            manager.connect_server("test_echo")  # expected if the mock server can't start

        # Disconnect and remove
        manager.disconnect_server("test_echo")
        manager.remove_server("test_echo")
        assert manager.list_servers() == []
