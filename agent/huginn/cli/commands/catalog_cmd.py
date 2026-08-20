"""Catalog 命令 —— 统一接入项的清单/启停/卸载 (CLI 侧).

与 API 端 (`routes/catalog.py`) 等价但离线可用: 复用各类注册表做发现,
不依赖正在跑的 server. 启停/卸载委托 `catalog.reconcile` 落地到注册表.

`ponytail:` CLI 是逐次独立进程, 每次调用内存不共享 —— tool 禁用靠 reconcile
的进程内 stash 复原, 所以跨进程 re-enable 会返回 unsupported. 跨进程的持久
启停语义在 server 运行时 (PATCH /catalog) 才有意义; 这里的 enable/disable
主要供单次交互进程内排查与验证.
"""
from __future__ import annotations

import asyncio

import click
from rich.table import Table

from huginn.catalog.manager import CatalogManager
from huginn.catalog.reconcile import apply_enabled
from huginn.cli.context import console

# origin 显示名, 便于人类阅读.
ORIGIN_LABEL = {
    "builtin": "built-in",
    "dirs": "dirs",
    ".mcp.json": ".mcp.json",
    "config": "config",
    "api": "api",
    "runtime": "runtime",
}


def _mcp_manager():
    """构建一个离线的 MCPClientManager, 载入 config 里的 server 清单."""
    from huginn.mcp_client import MCPClientManager

    mgr = MCPClientManager()
    with _quiet():
        mgr.load_from_huginn_config()
    return mgr


def _plugins_dir():
    """插件接入目录 (与 loader 同源, 覆盖 manifest/metadata/SKILL)."""
    from huginn.plugins.loader import DEFAULT_PLUGINS_DIR

    return DEFAULT_PLUGINS_DIR


import contextlib


@contextlib.contextmanager
def _quiet():
    """临时压掉 MCP auto-load 的 INFO 日志, 别让 CLI 输出被刷屏."""
    import logging

    logger = logging.getLogger("huginn.mcp_client")
    lvl = logger.level
    try:
        logger.setLevel(logging.ERROR)
        yield
    finally:
        logger.setLevel(lvl)


def _entries():
    """发现各注册表, 返回归一后的清单 (sorted by kind/name)."""
    cm = CatalogManager()
    cm.discover_all(mcp_manager=_mcp_manager(), plugins_dir=_plugins_dir())
    return cm


def _print_table(cm):
    rows = cm.list()
    if not rows:
        console.print("[yellow]No catalog entries found.[/yellow]")
        return
    table = Table(title="Catalog", show_lines=False)
    table.add_column("ID", style="cyan")
    table.add_column("Kind", style="bold")
    table.add_column("Name")
    table.add_column("Origin", style="magenta")
    table.add_column("Enabled", justify="center")
    for e in rows:
        enabled = "[green]on[/green]" if e.enabled else "[red]off[/red]"
        table.add_row(
            e.id,
            e.kind,
            e.name,
            ORIGIN_LABEL.get(e.origin, e.origin),
            enabled,
        )
    console.print(table)
    console.print(f"\n[dim]{len(rows)} entr[/dim]" + ("[dim]y[/dim]" if len(rows) == 1 else "[dim]ies[/dim]"))


@click.group(name="catalog")
def catalog() -> None:
    """List and manage the unified registration catalog."""


@catalog.command("list")
def catalog_list() -> None:
    """Show all unified registration entries (tools/skills/mcp/prompt)."""
    _print_table(_entries())


def _resolve(cm, entry_id: str):
    e = cm.get(entry_id)
    if e is None:
        console.print(f"[red]Unknown entry:[/red] {entry_id}")
        raise click.Abort()
    return e


def _apply(cm, entry_id: str, enabled: bool):
    e = _resolve(cm, entry_id)
    cm.set_enabled(entry_id, enabled)
    mgr = _mcp_manager() if e.kind == "mcp" else None
    result = asyncio.run(apply_enabled(e, enabled, mgr))
    if result == "noop":
        # prompt/skill/model 这类暂不支持落地, 只改追踪标记.
        console.print(f"[dim]{e.kind} has no disable semantics; set tracking only.[/dim]")
        return
    state = "enabled" if enabled else "disabled"
    console.print(f"[green]'{entry_id}' {state}[/green] (result={result})")


@catalog.command("enable")
@click.argument("entry_id")
def catalog_enable(entry_id: str) -> None:
    """Enable an entry (re-connect MCP, restore tool)."""
    _apply(_entries(), entry_id, True)


@catalog.command("disable")
@click.argument("entry_id")
def catalog_disable(entry_id: str) -> None:
    """Disable an entry (disconnect MCP, unregister tool)."""
    _apply(_entries(), entry_id, False)


@catalog.command("delete")
@click.argument("entry_id")
def catalog_delete(entry_id: str) -> None:
    """Uninstall an entry (disable first, then drop from tracking)."""
    cm = _entries()
    _resolve(cm, entry_id)
    _apply(cm, entry_id, False)
    cm.uninstall(entry_id)
    console.print(f"[green]Uninstalled '{entry_id}'.[/green]")