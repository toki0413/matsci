"""研究管线诊断工具面的**单一薄控制点**.

对照 `docs/capabilities-and-registration-spec.md` 的注册契约:
  - 原则: **不重复造注册表** —— LLM 可见工具 schema 应来自 `ToolRegistry`(统一 Catalog 控制面)。
  - 本模块是 spec 认可的"薄控制层": 对刻意轻量、不拉重工具栈的研究管线,
    把所有域诊断工具**归一到同一规范 schema 形状**, 并经"工具名"**回退到 ToolRegistry**。
  - 任何 (研究管线侧的) 新工具面**必须**经此解析, 不得内联再造 schema 字典
    —— 这正是测试 `tests/test_tool_surface_unification.py` 盯住的不变量。

输入:
  - ``str``        : ToolRegistry 中已注册工具名 → schema 取 `get_all_schemas()`, 调用走 `ToolRegistry.get(name).call`.
  - ``dict``       : 精简逃生口 `{"tool": schema, "handle": fn}` → 强制规范形状后原样挂载.
输出: ``(schemas, handlers)``, schemas 均为规范形状, 供 `client.chat.completions.create(tools=...)` 使用.
"""
from __future__ import annotations

from typing import Callable


def canonical_tool_shape(name: str, description: str, parameters: dict | None = None) -> dict:
    """规范工具 schema 形状 —— 与 ToolRegistry.get_all_schemas() 输出一致.

    单一形状定义: 任何研究管线工具面都必须落成这个形状, 避免各载体私自变体。
    """
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description or "",
            "parameters": parameters if isinstance(parameters, dict)
            else {"type": "object", "properties": {}, "additionalProperties": False},
        },
    }


def _try_registry() -> bool:
    """ToolRegistry 是否可导入(重栈未必在轻量环境). 不可用则仅支持 dict 逃生口. """
    try:
        import huginn.tools.registry as _r  # noqa: F401
        return True
    except Exception:  # noqa: BLE001 — ToolRegistry 不可用(轻量环境)则仅支持 dict 逃生口
        return False


def _registry_tool_schema(name: str) -> dict:
    from huginn.tools.registry import ToolRegistry
    for s in ToolRegistry.get_all_schemas():
        if s.get("function", {}).get("name") == name:
            return s
    raise KeyError(f"tool '{name}' 不在 ToolRegistry 中 (无法解析 schema)")


def _registry_call(name: str) -> Callable[[dict], str]:
    from huginn.tools.registry import ToolRegistry

    def _call(a: dict) -> str:
        tool = ToolRegistry.get(name)
        if tool is None:
            return f'{{"error": "unknown tool {name}"}}'
        res = tool.call(**a)
        return res if isinstance(res, str) else str(res)
    return _call


def resolve_diagnostic_tools(items) -> tuple[list[dict], dict[str, Callable[[dict], str]]]:
    """把研究管线诊断工具面归一到规范形状.

    `items`: None / list[str] / list[dict{"tool","handle"}] / 二者混排。
    返回 (schemas, handlers)；空输入返回 ([], {})。
    """
    schemas: list[dict] = []
    handlers: dict[str, Callable[[dict], str]] = {}
    registry_ok = _try_registry()
    for it in items or []:
        if isinstance(it, str):  # ToolRegistry 工具名
            schemas.append(_registry_tool_schema(it))
            handlers[it] = _registry_call(it)
            continue
        t = it.get("tool", {})
        fn = t.get("function", {})
        name = fn.get("name")
        if not name:
            raise ValueError("诊断工具 dict 缺少 'tool.function.name'")
        schemas.append(canonical_tool_shape(name, fn.get("description", ""), fn.get("parameters")))
        handlers[name] = it["handle"]
    # 若环境无 ToolRegistry 且只有 dict 逃生口, 也正常解析(registry_ok 只是旁路说明)
    return schemas, handlers


# ── MCP 面接线: 把研究管线诊断工具装箱进 ToolRegistry ───────────────────────
# 使 `CapabilityRegistry.scan_tool_registry() → mcp_export` 这条链能看到它们
# (否则域科学计算工具只在 run_research_program 的 dict 逃生口里, MCP tools/list 看不见).


def register_diagnostic_tools(diagnostic_tools) -> list[str]:
    """把研究管线诊断工具注册进 ToolRegistry, 返回注册名列表.

    接线链: ToolRegistry → CapabilityRegistry.scan_tool_registry()(自动装箱)
    → mcp_export (tools/list + tools/call)。注册后同一工具面既服务深研管线
    (resolve_diagnostic_tools 的 str 回退), 也服务 MCP 外部调用。
    重依赖 (HuginnTool 基类) lazy —— 本函数在轻量环境无人调用时零开销。
    """
    from huginn.core_types import ToolResult
    from huginn.tools.base import HuginnTool
    from huginn.tools.registry import ToolRegistry

    class _DictHandleTool(HuginnTool):
        """把 {schema, handle} 诊断工具装箱成 HuginnTool (MCP 导出/统一注册面用)."""

        category = "sim"
        read_only = True
        active = True

        def __init__(self, *, name: str, description: str,
                     parameters_schema: dict, handle: Callable) -> None:
            self.name = name
            self.description = description or "域诊断科学计算工具 (runtime-mount)"
            self._parameters_schema = parameters_schema
            self._handle = handle
            super().__init__()

        def is_available(self) -> bool:
            return True

        @property
        def input_json_schema(self) -> dict:
            return self._parameters_schema or {"type": "object", "properties": {}}

        async def call(self, args, context=None):
            try:
                res = self._handle(dict(args or {}))
                return ToolResult(success=True, data=res, metadata={})
            except Exception as exc:  # noqa: BLE001 — 工具边界收敛, 不上抛
                return ToolResult(success=False, error=f"{type(exc).__name__}: {exc}",
                                  metadata={})

    names: list[str] = []
    for it in diagnostic_tools or []:
        fn = it["tool"].get("function", {})
        name = fn.get("name")
        if not name:
            continue
        ToolRegistry.register(_DictHandleTool(
            name=name,
            description=fn.get("description", ""),
            parameters_schema=fn.get("parameters"),
            handle=it["handle"],
        ))
        names.append(name)
    return names