"""MCP 命令白名单校验的回归测试.

覆盖三个来源优先级 (env > config > 默认) 与非法命令阻断. 保持最小, 只测
get_mcp_allowed_commands / validate_mcp_command 的取舍行为.
"""

from __future__ import annotations

import pytest

import huginn.mcp_client as mc


def test_default_allowlist_applies_when_no_env_or_config():
    # 未设 env / 未走 get_settings 时退回内置默认
    assert "python" in mc.get_mcp_allowed_commands()
    assert mc.validate_mcp_command("python") == "python"


def test_env_override_takes_priority(monkeypatch):
    monkeypatch.setenv("HUGINN_MCP_ALLOWED_COMMANDS", "node,npx")
    assert mc.get_mcp_allowed_commands() == {"node", "npx"}
    with pytest.raises(ValueError, match="not allowed"):
        mc.validate_mcp_command("python")


def test_illegal_command_is_rejected():
    with pytest.raises(ValueError, match="not allowed"):
        mc.validate_mcp_command("rm")


# ── MCP 配置序列化脱敏 ────────────────────────────────────────────────

def test_mask_mcp_config_redacts_nested_secrets():
    raw = {
        "command": "npx",
        "args": ["-y", "notes"],
        "url": "https://example.com",
        "env": {"API_KEY": "sk-live", "PATH": "/usr/bin"},
        "headers": {"Authorization": "Bearer abc", "X-Custom": "ok"},
    }
    out = mc.mask_mcp_config(raw)
    # 非敏感字段原样保留
    assert out["command"] == "npx"
    assert out["url"] == "https://example.com"
    # 嵌套 env / headers 里的敏感键被掩码
    assert out["env"]["API_KEY"] == "***"
    assert out["env"]["PATH"] == "/usr/bin"
    assert out["headers"]["Authorization"] == "***"
    assert out["headers"]["X-Custom"] == "ok"


def test_mask_mcp_config_keeps_original_untouched():
    raw = {"env": {"TOKEN": "super-secret"}}
    _ = mc.mask_mcp_config(raw)
    assert raw["env"]["TOKEN"] == "super-secret"
    assert mc.mask_mcp_config(None) == {}


def test_list_servers_masks_config():
    mgr = mc.MCPClientManager()
    mgr.register_server("notes", {
        "command": "npx",
        "env": {"MY_TOKEN": "abc123"},
    })
    servers = mgr.list_servers()
    assert servers[0]["config"]["env"]["MY_TOKEN"] == "***"
    assert servers[0]["config"]["command"] == "npx"