"""工作流的对外"MCP 码头" — 把命名工作流导出成标准 MCP server.

与 ``capabilities/mcp_export`` 对称: 能力有码头 (capabilities-mcp), 工作流也给
一个码头 (workflows-mcp), 让任意 MCP host 能:
  - ``workflow_manifest``  罗列全部命名工作流 (含参数 schema)
  - ``view_workflow``      导出某工作流的单文件定义 (可搬运/再导入)
  - ``render_workflow``    渲染某模板的 stage 骨架 (纯描述, 不执行)

安全默认: **全只读** — 只暴露"工作流定义/模板", 不触发任何真实计算/工具执行,
符合"把流程分享给生态"的定位; 陌生 host 拿到的是可搬运的蓝图, 不是执行权.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from typing import Any

from huginn.workflows.registry import WorkflowRegistry

logger = logging.getLogger(__name__)

_EMPTY_OBJECT_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}


def _as_mcp_tool(name: str, description: str, schema: dict[str, Any] | None) -> Any:
    from mcp.types import Tool

    return Tool(name=name, description=description, inputSchema=schema or _EMPTY_OBJECT_SCHEMA)


class WorkflowMCPBackend:
    """把 WorkflowRegistry 包装成只读 MCP 后端."""

    def __init__(self) -> None:
        self._registry = WorkflowRegistry
        WorkflowRegistry.register_builtin_templates()

    # ── 清单 ────────────────────────────────────────────────────────

    def manifest(self) -> list[dict[str, Any]]:
        items = self._registry.manifest()
        return [
            {"name": it["name"], "kind": it["kind"], "category": it["category"],
             "description": it["description"],
             "n_stages": it["n_stages"], "n_subtasks": it["n_subtasks"],
             "params": it["params"]}
            for it in items
        ]

    def mcp_tools(self) -> list[Any]:
        return [
            _as_mcp_tool(
                "workflow_manifest",
                "罗列全部命名工作流(含类型/规模/参数 schema)",
                {"type": "object", "properties": {}},
            ),
            _as_mcp_tool(
                "view_workflow",
                "导出某工作流的单文件定义, 可搬运/再导入",
                {"type": "object", "properties": {
                    "name": {"type": "string", "description": "工作流名"},
                }, "required": ["name"]},
            ),
            _as_mcp_tool(
                "render_workflow",
                "渲染某模板的 stage 骨架(纯描述, 不执行)",
                {"type": "object", "properties": {
                    "name": {"type": "string", "description": "模板工作流名"},
                    "params": {"type": "object", "description": "模板参数", "default": {}},
                }, "required": ["name"]},
            ),
        ]

    def contains(self, name: str) -> bool:
        return any(t.name == name for t in self.mcp_tools())

    # ── 调用 ────────────────────────────────────────────────────────

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        args = arguments or {}
        if name == "workflow_manifest":
            return {"success": True, "data": self.manifest(), "error": "", "code": "ok"}
        if name == "view_workflow":
            wf_name = args.get("name")
            try:
                pkg = self._registry.export(wf_name)
            except KeyError:
                return {"success": False, "error": f"workflow not found: {wf_name}",
                        "data": None, "code": "not_found"}
            return {"success": True, "data": pkg, "error": "", "code": "ok"}
        if name == "render_workflow":
            wf_name = args.get("name")
            params = args.get("params") or {}
            try:
                stages = self._registry.render_stages(wf_name, **params)
            except (ValueError, TypeError) as exc:
                return {"success": False, "error": f"cannot render '{wf_name}': {exc}",
                        "data": None, "code": "render_error"}
            return {"success": True, "data": {"name": wf_name, "n_stages": len(stages),
                                               "stages": stages}, "error": "", "code": "ok"}
        return {"success": False, "error": f"unknown workflow command: {name}",
                "data": None, "code": "not_found"}


def build_server(backend: WorkflowMCPBackend, server_name: str = "huginn-workflows") -> Any:
    from mcp.server import Server

    server = Server(server_name)

    @server.list_tools()
    async def _list() -> Any:
        return backend.mcp_tools()

    @server.call_tool()
    async def _call(name: str, arguments: dict[str, Any]) -> Any:
        from mcp.types import CallToolResult, TextContent

        out = await backend.call(name, arguments or {})
        text = json.dumps(out, ensure_ascii=False, default=str)
        return CallToolResult(
            isError=not out.get("success"),
            content=[TextContent(type="text", text=text)],
        )

    return server


def server(server_name: str = "huginn-workflows") -> Any:
    return build_server(WorkflowMCPBackend(), server_name=server_name)


async def serve_stdio(server_: Any) -> None:
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (rs, ws):
        await server_.run(rs, ws, server_.create_initialization_options())


async def serve_http(server_: Any, host: str, port: int) -> None:
    from mcp.server.http import streamable_http_server
    from uvicorn import Config
    from uvicorn import Server as UvicornServer

    app = streamable_http_server(server_)
    await UvicornServer(Config(app=app, host=host, port=port, log_level="info")).serve()


def list_json() -> str:
    backend = WorkflowMCPBackend()
    return json.dumps(backend.manifest(), ensure_ascii=False, indent=2)


def main(argv: list[str] | None = None) -> None:
    """CLI: ``python -m huginn.workflows.mcp_export ...``."""
    parser = argparse.ArgumentParser(
        prog="huginn-workflows-mcp", description="把 Huginn 命名工作流导出成 MCP server")
    parser.add_argument("--transport", choices=("stdio", "http"), default="stdio",
                        help="stdio 本机 MCP host / http 远程调用")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--list", action="store_true", help="只打印工作流清单退出")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    if args.list:
        print(list_json())
        return
    srv = server()
    try:
        if args.transport == "stdio":
            asyncio.run(serve_stdio(srv))
        else:
            asyncio.run(serve_http(srv, args.host, args.port))
    except KeyboardInterrupt:
        logger.info("shutting down workflow MCP server")


if __name__ == "__main__":
    main()