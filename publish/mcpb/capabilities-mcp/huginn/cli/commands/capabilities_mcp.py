"""CLI: 把能力集装箱导出成 MCP server (对外的"码头").

参考用法:
    huginn capabilities-mcp --list                    # 打印可分享能力清单
    huginn capabilities-mcp --transport stdio         # 本机 MCP host 接走能力
    huginn capabilities-mcp --transport http --port 8000 --allow-write
"""

from __future__ import annotations

import asyncio

import click

from huginn.capabilities.registry import CapabilityRegistry


@click.command("capabilities-mcp")
@click.option(
    "--transport",
    type=click.Choice(["stdio", "http"]),
    default="stdio",
    help="stdio=本机 MCP host; http=远程调用(共享经济对外服务形态)",
)
@click.option("--host", default="127.0.0.1", show_default=True, help="http 监听地址")
@click.option("--port", default=8000, type=int, show_default=True, help="http 端口")
@click.option(
    "--allow-write",
    is_flag=True,
    default=False,
    help="同时暴露非只读(写/破坏性)能力. 默认只读, 安全第一.",
)
@click.option(
    "--list",
    "list_only",
    is_flag=True,
    default=False,
    help="只打印能力清单(OpenAI function JSON)后退出.",
)
def capabilities_mcp(transport, host, port, allow_write, list_only):
    """把能力集装箱导出成 MCP server, 让任意 MCP host 直接调用."""
    # 先装箱: 加载全部激活工具为原子能力 + 注册组合能力; 否则清单是空的.
    from huginn.capabilities.mcp_export import (
        list_json,
        serve_http,
        serve_stdio,
        server,
    )
    from huginn.tools import register_capability_tools
    from huginn.tools.registry import ToolRegistry

    # 兜底: 若启动时 main() 未预注册工具 (单独跑本命令), 先装配工具池,
    # 否则能力清单里只有显式组合能力, 原子能力为空.
    if not ToolRegistry.list_tools():
        from huginn.tools import register_all_tools

        register_all_tools(None)

    registered = register_capability_tools(None)
    CapabilityRegistry.scan_tool_registry()
    if not registered and not CapabilityRegistry.list_capabilities():
        click.echo("No capabilities available to export (tool registration failed fixture).")
        return

    if list_only:
        click.echo(list_json(allow_write=allow_write))
        click.echo(
            f"\n# {len(CapabilityRegistry.list_capabilities())} capabilities "
            f"(read-only only: {not allow_write})",
            err=True,
        )
        return

    n_caps = len(CapabilityRegistry.list_capabilities())
    srv = server(allow_write=allow_write)
    click.echo(f"Serving {n_caps} capabilities via {transport}", err=True)
    asyncio.run(
        serve_stdio(srv) if transport == "stdio" else serve_http(srv, host, port)
    )
