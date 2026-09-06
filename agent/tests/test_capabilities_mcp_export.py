"""能力"MCP 码头"测试 — 协议无关桥 + 安全过滤 + server 组装.

不启动真实 MCP 传输/子进程; 只测:
- as_mcp_tool / as_openai_function 的映射正确性
- CapabilityMCPBackend: 默认只暴露只读能力, allow_write 才放开
- backend.call: 找不到/隐藏能力时报错, 找到时分派到能力并 JSON-safe
- build_server: list_tools/call_tool 装饰器已接上 (用注入的假后端验 handler)
"""

from __future__ import annotations

import asyncio

from huginn.core_types import ToolContext
from huginn.tools.base import HuginnTool


def _ctx() -> ToolContext:
    return ToolContext(session_id="test", workspace=".")


# ── 协议无关桥: 映射 ────────────────────────────────────────────────────


class TestMCPBridge:
    def test_as_mcp_tool_maps_fields(self):
        from huginn.capabilities.mcp_export import as_mcp_tool

        tool = as_mcp_tool(
            "lit_search",
            "search literature",
            {"type": "object", "properties": {"q": {"type": "string"}}},
        )
        assert tool.name == "lit_search"
        assert tool.description == "search literature"
        assert tool.inputSchema["properties"]["q"]["type"] == "string"

    def test_as_mcp_tool_empty_schema_fallback(self):
        from huginn.capabilities.mcp_export import as_mcp_tool

        tool = as_mcp_tool("no_schema", None, None)
        assert tool.inputSchema == {"type": "object", "properties": {}}

    def test_as_openai_function_shape(self):
        from huginn.capabilities.mcp_export import as_openai_function

        fn = as_openai_function("cap_x", "desc", {"type": "object"})
        assert fn["type"] == "function"
        fun = fn["function"]
        assert fun["name"] == "cap_x"
        assert fun["description"] == "desc"
        assert fun["parameters"] == {"type": "object"}


# ── 后端: 安全过滤 + 分派 ────────────────────────────────────────────────


class _ROTool(HuginnTool):
    name = "ro_tool"
    description = "read only"
    read_only = True
    input_schema = None

    async def call(self, args, context):
        from huginn.core_types import ToolResult

        return ToolResult(data={"echo": args}, success=True)


class _WriteTool(HuginnTool):
    name = "write_tool"
    description = "writes stuff"
    read_only = False
    input_schema = None

    async def call(self, args, context):
        from huginn.core_types import ToolResult

        return ToolResult(data={"wrote": args}, success=True)


def _isolated_backend(**kw):
    from huginn.capabilities.intents import AtomicCapability
    from huginn.capabilities.mcp_export import CapabilityMCPBackend
    from huginn.capabilities.registry import CapabilityRegistry

    CapabilityRegistry.clear()
    CapabilityRegistry.register(AtomicCapability(_ROTool()))
    CapabilityRegistry.register(AtomicCapability(_WriteTool()))
    return CapabilityMCPBackend(registry=CapabilityRegistry, **kw)


class TestCapabilityMCPBackend:
    def _clear(self):
        from huginn.capabilities.registry import CapabilityRegistry

        CapabilityRegistry.clear()

    def test_default_exposes_read_only_only(self):
        backend = _isolated_backend()
        names = {m["name"] for m in backend.manifest()}
        assert "ro_tool" in names
        assert "write_tool" not in names
        self._clear()

    def test_allow_write_exposes_everything(self):
        backend = _isolated_backend(allow_write=True)
        names = {m["name"] for m in backend.manifest()}
        assert {"ro_tool", "write_tool"} <= names
        self._clear()

    def test_call_success(self):
        backend = _isolated_backend()
        out = asyncio.run(backend.call("ro_tool", {"q": "hi"}))
        assert out["success"] is True
        assert out["data"]["echo"]["q"] == "hi"
        self._clear()

    def test_call_hidden_write_tool_is_refused(self):
        backend = _isolated_backend()  # write_tool hidden
        out = asyncio.run(backend.call("write_tool", {"a": 1}))
        assert out["success"] is False
        assert out["code"] == "not_found"
        self._clear()

    def test_call_unknown(self):
        backend = _isolated_backend()
        out = asyncio.run(backend.call("nope", {}))
        assert out["success"] is False
        assert out["code"] == "not_found"
        self._clear()


# ── server 组装 ──────────────────────────────────────────────────────────


class _FakeServerBackend:
    """最小假后端, 独立于 mcp SDK 验证 build_server 接线的 handler 语义."""

    def __init__(self):
        self._tools = [("cap_x", "desc", {"type": "object"})]

    def manifest(self):
        return [
            {"name": "cap_x", "description": "desc", "input_schema": {"type": "object"}}
        ]

    def mcp_tools(self):
        from huginn.capabilities.mcp_export import as_mcp_tools

        return as_mcp_tools(self.manifest())

    def openai_functions(self):
        from huginn.capabilities.mcp_export import as_openai_functions

        return as_openai_functions(self.manifest())

    def contains(self, name):
        return name == "cap_x"

    async def call(self, name, arguments):
        return {
            "success": True,
            "data": {"echo": arguments},
            "error": None,
            "code": "ok",
        }


class TestBuildServer:
    def test_build_server_wires_handlers(self):
        from huginn.capabilities.mcp_export import build_server

        srv = build_server(_FakeServerBackend(), server_name="ut")
        # 装饰器已注册, 且成功构造出 mcp SDK 的 Server 实例 (import 链路通).
        assert type(srv).__name__ == "Server"

    def test_list_tools_handler_returns_definitions(self):
        from huginn.capabilities.mcp_export import list_tools_handler

        tools = asyncio.run(list_tools_handler(_FakeServerBackend()))
        assert len(tools) == 1
        assert tools[0].name == "cap_x"

    def test_call_tool_handler_success(self):
        from huginn.capabilities.mcp_export import call_tool_handler

        result = asyncio.run(call_tool_handler("cap_x", {"q": 1}, _FakeServerBackend()))
        assert result.isError is False
        text = result.content[0].text
        assert '"echo"' in text

    def test_call_tool_handler_hidden(self):
        from huginn.capabilities.mcp_export import call_tool_handler

        result = asyncio.run(call_tool_handler("secret", {}, _FakeServerBackend()))
        assert result.isError is True
        assert "not found" in result.content[0].text


def test_export_module_guards_return_json_safe():
    from huginn.capabilities.mcp_export import _jsonable

    assert _jsonable({"a": [1, 2]}) == {"a": [1, 2]}
    assert isinstance(_jsonable(object()), str)  # 不可序列化回落成字符串
