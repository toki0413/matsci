"""制度化闭环：huginn 消费仓库根 `.mcp.json`（标准 MCP 配置）。

IDE/桌面读 `.mcp.json` 连外部 MCP（如 flint-chart），huginn 的 lifespan 现在
也消费同一份文件，把这些 server 拉起并注册进工具池。这道门禁保证：
- 根 `.mcp.json` 里的 mcpServers 能被正确解析成 MCPServerConfig；
- 文件缺失 / 非法 JSON / 单条配置非法时 best-effort 降级，不抛错。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from huginn.lifespan import _load_mcp_json_servers

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_loads_repo_root_mcp_json_into_configs():
    servers = _load_mcp_json_servers(REPO_ROOT)
    by_name = {name: cfg for name, cfg in servers}

    # 根 .mcp.json 至少声明了 flint-chart（npx -y flint-chart-mcp）
    assert "flint-chart" in by_name
    cfg = by_name["flint-chart"]
    assert cfg.command == "npx"
    assert cfg.args == ["-y", "flint-chart-mcp"]
    assert cfg.transport == "stdio"


def test_missing_file_returns_empty():
    assert _load_mcp_json_servers(REPO_ROOT / "does_not_exist_XYZ") == []


def test_invalid_json_is_best_effort(tmp_path):
    bad = tmp_path / ".mcp.json"
    bad.write_text("{ not valid json", encoding="utf-8")
    assert _load_mcp_json_servers(tmp_path) == []


def test_invalid_entry_is_skipped(tmp_path):
    cfg = tmp_path / ".mcp.json"
    cfg.write_text(
        '{"mcpServers": {"ok": {"command": "python", "args": []}, '
        '"bad": "not-a-dict"}}',
        encoding="utf-8",
    )
    servers = _load_mcp_json_servers(tmp_path)
    assert [n for n, _ in servers] == ["ok"]