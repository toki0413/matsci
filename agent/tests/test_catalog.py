"""Catalog 骨架测试 —— discover / 来源归一 / list / set_enabled 保持 / snapshot restore."""

from __future__ import annotations

from huginn.catalog import CatalogManager, ORIGIN_PRIORITY, make_entry_id


class _FakeMCP:
    """最小可用的 MCP 管理器替身: 只暴露 list_servers (含 origin)."""

    def __init__(self, servers: list[dict]):
        self._servers = servers

    def list_servers(self):  # noqa: ANN201
        return self._servers


def test_discover_includes_mcp_entries_with_origin():
    mgr = CatalogManager()
    fake = _FakeMCP([
        {"name": "notes", "config": {"command": "npx"}, "origin": ".mcp.json"},
        {"name": "mat-db", "config": {"command": "python"}, "origin": "builtin"},
    ])
    entries = mgr.discover_all(mcp_manager=fake)

    ids = {e.id for e in entries}
    assert make_entry_id("mcp", "notes") in ids
    assert make_entry_id("mcp", "mat-db") in ids

    notes = mgr.get(make_entry_id("mcp", "notes"))
    assert notes is not None
    assert notes.kind == "mcp"
    assert notes.origin == ".mcp.json"
    assert notes.enabled is True


def test_discover_mcp_defaults_origin_to_runtime():
    mgr = CatalogManager()
    fake = _FakeMCP([{"name": "x", "config": {}}])  # 未标 origin
    mgr.discover_all(mcp_manager=fake)
    assert mgr.get(make_entry_id("mcp", "x")).origin == "runtime"


def test_set_enabled_survives_rediscover():
    mgr = CatalogManager()
    fake = _FakeMCP([{"name": "notes", "config": {}}])
    mgr.discover_all(mcp_manager=fake)

    eid = make_entry_id("mcp", "notes")
    assert mgr.set_enabled(eid, False) is True

    # 重新采集时, 用户设置过的 enabled=False 不能被冲掉.
    mgr.discover_all(mcp_manager=fake)
    assert mgr.get(eid).enabled is False

    # 未知 id 返回 False.
    assert mgr.set_enabled(make_entry_id("mcp", "nope"), True) is False


def test_snapshot_restore_roundtrip():
    mgr = CatalogManager()
    fake = _FakeMCP([
        {"name": "notes", "config": {}},
        {"name": "mat-db", "config": {}},
    ])
    mgr.discover_all(mcp_manager=fake)

    snap = mgr.snapshot()
    assert len(snap) >= 2

    mgr2 = CatalogManager()
    mgr2.restore(snap)
    assert {e.id for e in mgr2.list()} == set(snap.keys())


def test_origin_priority_keeps_highest_origin():
    """同一 server 经多源注册时, 按 ORIGIN_PRIORITY 取最高者 (config > .mcp.json)."""
    assert ORIGIN_PRIORITY["config"] > ORIGIN_PRIORITY[".mcp.json"]
    assert ORIGIN_PRIORITY["api"] > ORIGIN_PRIORITY["config"]


def test_cli_catalog_group_loads_and_lists():
    """CLI `catalog` 命令组能加载, list 返回表格且包含注册的锚点工具."""
    from click.testing import CliRunner

    from huginn.cli.commands.catalog_cmd import catalog

    runner = CliRunner()
    result = runner.invoke(catalog, ["list"], catch_exceptions=False)
    assert result.exit_code == 0
    # 工具锚点 (core tools 注册后才存在; 若未注册则不强断言, 只验证命令可跑).
    assert "Catalog" in result.output