"""CLI: 把命名工作流导出成 MCP server (工作流的"码头").

参考用法:
    huginn workflows-mcp --list                    # 打印可分享工作流清单
    huginn workflows-mcp --transport stdio         # 本机 MCP host 接走工作流定义
    huginn workflows-mcp --transport http --port 8001
"""
from __future__ import annotations

import asyncio

import click

from huginn.workflows.mcp_export import (
    list_json,
    server,
    serve_http,
    serve_stdio,
)


@click.command("workflows-mcp")
@click.option(
    "--transport",
    type=click.Choice(["stdio", "http"]),
    default="stdio",
    help="stdio=本机; http=远程调用(把工作流蓝图暴露给生态)",
)
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8001, type=int, show_default=True)
@click.option("--list", "list_only", is_flag=True, default=False,
              help="只打印可分享工作流清单后退出")
def workflows_mcp(transport, host, port, list_only):
    """把命名工作流导出成 MCP server, 让任意 MCP host 拿走蓝图."""
    from huginn.workflows.registry import WorkflowRegistry

    WorkflowRegistry.register_builtin_templates()
    n = len(WorkflowRegistry.list_names())
    if list_only:
        click.echo(list_json())
        click.echo(f"\n# {n} 个工作流可分享 (只读蓝图)", err=True)
        return
    srv = server()
    click.echo(f"Serving {n} workflows via {transport}", err=True)
    asyncio.run(serve_stdio(srv) if transport == "stdio" else serve_http(srv, host, port))