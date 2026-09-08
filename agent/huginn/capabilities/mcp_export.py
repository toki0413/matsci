"""能力集装箱的对外"MCP 码头" — 把能力清单导出成标准 MCP server.

背景 (共享经济 / 生态融入): 能力集装箱标准化了"箱子" (统一 run() 契约 +
注册表 + 组合), 但一个人作品要融入更大的 agent 生态, 还差一个"码头":
让 Claude Desktop / Cursor / Trae / 任意 MCP host 能把我们整箱能力直接
召唤走, 而不是只能 import 我们的 Python 包.

本模块就是那个码头:
- 协议无关桥: ``as_mcp_tool`` / ``as_openai_function`` 把能力清单的某一项
  序列化成标准 tool 描述 (MCP JSON Schema / OpenAI function). 这样后续再加
  A2A / HTTP 码头, 只写薄薄一段适配, 不用重来.
- MCP server: ``CapabilityMCPBackend`` + :func:`build_server` 把当前注册表里
  的所有能力暴露成 ``tools/list`` + ``tools/call``, 支持 stdio / HTTP-streamable
  传输.
- 安全默认: **默认只暴露只读能力** (``allow_write=False``). 写/破坏性能力要
  显式 ``--allow-write`` 才会对远端 host 可见 — 跟凭个人能力共享的定位一致,
  不让陌生 host 一来就动磁盘/跑命令.

调用方可通过 ``context_factory`` 注入带 memory_manager 等运行时字段的
:class:`~huginn.core_types.ToolContext`, 让远程调用挂进自有 agent 的上下文.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from collections.abc import Callable
from typing import Any

from huginn.capabilities.registry import CapabilityRegistry
from huginn.core_types import ToolContext

logger = logging.getLogger(__name__)

# 最少输入 schema: 能力未声明 input_schema 时兜底, 符合 MCP Tool 要求.
_EMPTY_OBJECT_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}


# ── 协议无关桥: 能力清单项 → 标准 tool 描述 ──────────────────────────────


def as_mcp_tool(
    name: str, description: str | None, input_schema: dict[str, Any] | None
) -> Any:
    """把能力描述映射成 MCP ``mcp.types.Tool``.

    返回类型标注为 Any 以便在未安装 mcp SDK 的环境里, 本模块仍可被
    import (调用端自行惰性加载). 实际安装时返回 mcp.types.Tool.
    """
    from mcp.types import Tool

    return Tool(
        name=name,
        description=description or "",
        inputSchema=input_schema or _EMPTY_OBJECT_SCHEMA,
    )


def as_mcp_tools(manifest: list[dict[str, Any]]) -> list[Any]:
    """批量: 能力清单 (CapabilityRegistry.manifest()) → MCP Tool 列表."""
    return [
        as_mcp_tool(
            name=m["name"],
            description=m.get("description"),
            input_schema=m.get("input_schema"),
        )
        for m in manifest
    ]


def as_openai_function(
    name: str, description: str | None, input_schema: dict[str, Any] | None
) -> dict[str, Any]:
    """把能力描述映射成 OpenAI function-calling 工具定义.

    让不装 MCP 客户端的普通 LLM 应用也能通过 HTTP 直接调用我们的能力.
    """
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description or "",
            "parameters": input_schema or _EMPTY_OBJECT_SCHEMA,
        },
    }


def as_openai_functions(manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        as_openai_function(
            name=m["name"],
            description=m.get("description"),
            input_schema=m.get("input_schema"),
        )
        for m in manifest
    ]


# ── 后端: 统一访问入口 + 安全过滤 + 能力分发 ──────────────────────────────


def _default_context_factory(_tool_name: str) -> ToolContext:
    """远端调用无自有上下文时兜底: 一个就地构造的空 ToolContext."""
    return ToolContext(session_id="mcp-export", workspace=os.getcwd())


class CapabilityMCPBackend:
    """把 CapabilityRegistry 包装成可被 MCP server 驱动的后端.

    负责: (1) 按安全策略过滤可见能力, (2) ``tools/list`` 的素材
    (定义), (3) ``tools/call`` 的分派 (经 CapabilityRegistry.run).
    """

    def __init__(
        self,
        *,
        allow_write: bool = False,
        context_factory: Callable[[str], ToolContext] = _default_context_factory,
        registry: type[CapabilityRegistry] | None = None,
    ) -> None:
        self._allow_write = allow_write
        self._context_factory = context_factory
        self._registry = registry or CapabilityRegistry
        # 启动时快照清单: 远端 host 视角的能力集应当稳定, 而不是每次调用在变.
        self._manifest = self._filtered_manifest()

    # ── 清单 ────────────────────────────────────────────────────────

    def _filtered_manifest(self) -> list[dict[str, Any]]:
        all_items = self._registry.manifest()
        if self._allow_write:
            return all_items
        return [m for m in all_items if m.get("read_only")]

    def manifest(self) -> list[dict[str, Any]]:
        return self._manifest

    def mcp_tools(self) -> list[Any]:
        return as_mcp_tools(self._manifest)

    def openai_functions(self) -> list[dict[str, Any]]:
        return as_openai_functions(self._manifest)

    # ── 调用 ────────────────────────────────────────────────────────

    def contains(self, name: str) -> bool:
        return any(m["name"] == name for m in self._manifest)

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """执行能力, 返回 JSON-safe 的 ``{success, data, error, code}``."""
        if not self.contains(name):
            return {
                "success": False,
                "error": f"capability not found or hidden: {name}",
                "code": "not_found",
            }
        context = self._context_factory(name)
        result = await self._registry.run(name, arguments or {}, context)
        return {
            "success": result.success,
            "data": _jsonable(result.data),
            "error": result.error,
            "code": result.code,
        }


def _jsonable(obj: Any) -> Any:
    """尽力把能力结果规整成 JSON-safe 结构 (失败时回退成字符串)."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (dict, list)):
        return obj
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:  # serialization 兜底 → 回退 str
            return str(obj)
    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()
        except Exception:  # serialization 兜底 → 回退 str
            return str(obj)
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:  # serialization 兜底 → 回退 str
            return str(obj)
    return str(obj)


# ── 组装 MCP server ──────────────────────────────────────────────────────


async def list_tools_handler(backend: CapabilityMCPBackend | Any) -> list[Any]:
    """MCP ``tools/list`` 语义: 返回后端过滤后的能力定义列表.

    独立为可测纯函数; build_server 里的装饰器只做薄转接.
    """
    return backend.mcp_tools()


async def call_tool_handler(
    name: str, arguments: dict[str, Any], backend: CapabilityMCPBackend | Any
) -> Any:
    """MCP ``tools/call`` 语义: 分派到能力, 结果规整成 CallToolResult.

    独立为可测纯函数, 返回 ``mcp.types.CallToolResult``.
    """
    from mcp.types import CallToolResult, TextContent

    if not backend.contains(name):
        return CallToolResult(
            isError=True,
            content=[
                TextContent(type="text", text=f"capability not found or hidden: {name}")
            ],
        )
    out = await backend.call(name, arguments or {})
    text = json.dumps(out, ensure_ascii=False, default=str)
    if out.get("success"):
        return CallToolResult(content=[TextContent(type="text", text=text)])
    return CallToolResult(
        isError=True,
        content=[TextContent(type="text", text=text)],
    )


def build_server(
    backend: CapabilityMCPBackend, server_name: str = "huginn-capabilities"
) -> Any:
    """把后端接成一个 mcp.server.Server (低层 API).

    ``tools/list`` 返回后端过滤后的能力; ``tools/call`` 分派到能力执行.
    函数返回 ``Any`` (惰性 import mcp), 让上层在未装 mcp 的环境也能 import.
    """
    from mcp.server import Server

    server = Server(server_name)

    @server.list_tools()
    async def _list() -> Any:
        return await list_tools_handler(backend)

    @server.call_tool()
    async def _call(name: str, arguments: dict[str, Any]) -> Any:
        return await call_tool_handler(name, arguments, backend)

    return server


# ── 传输入口 (stdio / http-streamable) ──────────────────────────────────


def server(
    *,
    allow_write: bool = False,
    context_factory: Callable[[str], ToolContext] = _default_context_factory,
    server_name: str = "huginn-capabilities",
) -> Any:
    """一步拿到已装配的 MCP server (供上层接传输用)."""
    backend = CapabilityMCPBackend(
        allow_write=allow_write, context_factory=context_factory
    )
    return build_server(backend, server_name=server_name)


async def serve_stdio(server_: Any) -> None:
    """stdio 传输: 让外部 MCP host 通过子进程 stdin/stdout 调用能力."""
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        await server_.run(
            read_stream,
            write_stream,
            server_.create_initialization_options(),
        )


def _http_app(server_: Any) -> Any:
    """按当前 mcp SDK 版本选 API 组装 streamable HTTP ASGI app (跨版本兼容)."""
    try:
        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
        return StreamableHTTPSessionManager(server_)
    except ImportError:
        pass
    try:
        from mcp.server.streamable_http import streamable_http_server
        return streamable_http_server(server_)
    except ImportError:
        pass
    from mcp.server.http import streamable_http_server
    return streamable_http_server(server_)


async def serve_http(server_: Any, host: str, port: int) -> None:
    """HTTP-streamable 传输: 让远程 MCP host 通过 HTTP POST 调用能力.

    建在 starlette/uvicorn 上, 便于 LAN / 内网共享 (共享经济的"对外服务"形态).
    """
    from uvicorn import Config
    from uvicorn import Server as UvicornServer

    app = _http_app(server_)
    cfg = Config(app=app, host=host, port=port, log_level="info")
    uvicorn_server = UvicornServer(cfg)
    await uvicorn_server.serve()


def list_json(*, allow_write: bool = False) -> str:
    """只打印能力清单 (OpenAI function 形态) 的 JSON, 供宿主侧预览/裁剪."""
    backend = CapabilityMCPBackend(allow_write=allow_write)
    return json.dumps(backend.openai_functions(), ensure_ascii=False, indent=2)


def _ensure_capabilities_registered() -> None:
    """确保原子能力来自已装配的工具池.

    ``python -m`` 单跑时没有 main() 的预注册, 工具池为空则能力清单是空的.
    defensively 装配: 装填全部激活工具 + 组合能力预设.
    """
    from huginn.tools import register_capability_tools
    from huginn.tools.registry import ToolRegistry

    if not ToolRegistry.list_tools():
        from huginn.tools import register_all_tools

        register_all_tools(None)
    register_capability_tools(None)


def main(argv: list[str] | None = None) -> None:
    """CLI: ``python -m huginn.capabilities.mcp_export ...``."""
    parser = argparse.ArgumentParser(
        prog="huginn-capabilities-mcp",
        description="把 Huginn 能力集装箱导出成 MCP server",
    )
    parser.add_argument(
        "--transport",
        choices=("stdio", "http"),
        default="stdio",
        help="传输方式: stdio(默认) 供本机 MCP host; http 供远程调用",
    )
    parser.add_argument("--host", default="127.0.0.1", help="http 传输监听地址")
    parser.add_argument("--port", type=int, default=8000, help="http 传输端口")
    parser.add_argument(
        "--allow-write",
        action="store_true",
        help="同时暴露非只读(写/破坏性)能力. 默认只暴露只读能力, 安全第一.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="只打印能力清单(OpenAI function JSON)后退出, 不启动 server",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    _ensure_capabilities_registered()

    if args.list:
        print(list_json(allow_write=args.allow_write))
        return

    srv = server(allow_write=args.allow_write)
    try:
        if args.transport == "stdio":
            asyncio.run(serve_stdio(srv))
        else:
            asyncio.run(serve_http(srv, args.host, args.port))
    except KeyboardInterrupt:
        logger.info("shutting down capability MCP server")


if __name__ == "__main__":
    main()
