"""工具面统一化不变量测试 —— 把"不重复造注册表"契约变成常驻机械检查.

对标 `capabilities-and-registration-spec.md`: 研究管线(轻量路径)的 LLM 可见工具
schema 必须来自 `huginn/research/tool_surface.resolve_diagnostic_tools` 这一**单一薄
控制点**, 遵循规范形状; 不得在各载体里内联再造 schema 字典(以往碎片化根因)。

覆盖:
  1. 规范形状与 ToolRegistry 输出一致 (type/function/name/description/parameters).
  2. resolve_diagnostic_tools 对 dict 逃生口与 (可用时) ToolRegistry 工具名都归一.
  3. run_research_program 使用该解析器, 而非内联造 schema (源码级不变量).
  4. 现有域后端 DIAGNOSTIC_TOOLS 条目都能被解析成规范形状, 不出现孤岛变体.

纯标准库/轻依赖, 零网络/零 LLM, 确定性。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from huginn.research.tool_surface import canonical_tool_shape, resolve_diagnostic_tools

_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def _fake_handle(a):
    return '{"harmonic_frac": 0.42}'


def test_canonical_shape_matches_toolregistry_output_form():
    """规范形状与 ToolRegistry.get_all_schemas() 的 openai 形态一致."""
    s = canonical_tool_shape("hodge_circulation", "desc", {"type": "object", "properties": {}})
    assert s["type"] == "function"
    f = s["function"]
    assert f["name"] == "hodge_circulation"
    assert f["description"] == "desc"
    assert f["parameters"]["type"] == "object"


def test_resolve_dict_escape_hatch():
    sch, handlers = resolve_diagnostic_tools([
        {"tool": canonical_tool_shape("toolA", "A"),
         "handle": _fake_handle},
    ])
    assert len(sch) == 1 and sch[0]["function"]["name"] == "toolA"
    assert "toolA" in handlers and handlers["toolA"]({}) == '{"harmonic_frac": 0.42}'


def test_resolve_normalizes_missing_parameters():
    """缺 parameters 的 dict 也被注入默认 object, 不产生畸形 schema."""
    sch, _ = resolve_diagnostic_tools([{"tool": canonical_tool_shape("toolB", "B"), "handle": _fake_handle}])
    assert sch[0]["function"]["parameters"] == {"type": "object", "properties": {}, "additionalProperties": False}


def test_program_sources_from_single_thin_control_point():
    """run_research_program 必须经由 resolve_diagnostic_tools, 不得内联再造 schema 字典."""
    src = (_ROOT / "huginn/research/program.py").read_text(encoding="utf-8")
    assert "from huginn.research.tool_surface import resolve_diagnostic_tools" in src
    assert "resolve_diagnostic_tools(diagnostic_tools)" in src
    # 不得出现内联 dict 拼接式的工具面 (旧式 `[t["tool"] for t in ...]` 形式已被替换)
    assert "t[\"tool\"] for t in" not in src


def test_backend_diagnostic_tools_resolve_and_no_island_variant():
    """域后端 DIAGNOSTIC_TOOLS 的每条都能经解析器归一成规范形状."""
    sys.path.insert(0, str(_EXAMPLES))
    import ai4s_backends as ab
    for domain, tools in ab.DIAGNOSTIC_TOOLS.items():
        sch, handlers = resolve_diagnostic_tools(tools)
        assert len(sch) == len(tools) == len(handlers)
        for s in sch:
            assert s["type"] == "function"
            assert s["function"]["name"] in handlers


def test_resolve_registry_by_name_when_available():
    """可用时支持 ToolRegistry 工具名 (best-effort; 无重注册表环境则跳过)."""
    try:
        from huginn.tools.registry import ToolRegistry
        names = ToolRegistry.list_tools()
    except Exception:  # noqa: BLE001 — 重栈不可用, 跳过注册表名路径
        return
    if not names:
        return  # 环境无工具可解析
    name = names[0]
    sch, handlers = resolve_diagnostic_tools([name])
    assert sch[0]["function"]["name"] == name
    assert name in handlers and callable(handlers[name])