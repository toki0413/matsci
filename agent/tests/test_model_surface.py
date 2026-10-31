"""模型面契约 —— 移植 dsh `tests/model-surface.spec.ts` 的度量原则.

dsh 把"模型能看到的全部文本"当成一份契约: 系统 prompt、工具定义、输入与错误逐条
收集, 检查引用合法、无退役/禁用词、`KNOWN_FAILURES` 必须为空(双向核对: 新漂移红,
已修复未摘除也红), 并给每个角色的 prompt + 工具定义一个**只许下调**的字符预算.

本文件是 huginn 的 **observe 阶段**移植:
- prompt 面走真实 `prompt_segments` 渲染 (按生产调用形状), 不造夹具;
- 工具面走真实 `ToolRegistry.get_all_schemas()` 的 `function` 定义 —— 与 LLM
  实际收到的字节一致 —— 并用 snapshot/restore 隔离全局注册表;
- `BUDGET` 为空 = 只测不判; 冻结后填上限, 上调预算须显式改本文件 (只减不增).

诚实边界:
- 只覆盖**已注册**工具与内置 prompt 段; 运行时按会话/部署拼入的 persona、工作区
  上下文、规则文本不计入 (随部署变化, 不可确定性度量).
- 只断言"模型面文本无占位词/无空描述", **不**暗示模型一定遵守 prompt.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys
import types

import pytest

_AGENT = pathlib.Path(__file__).resolve().parents[1]

# 与 test_prompt_contract 同款: 只给 __path__, 绕过 huginn.agent 的重依赖 __init__.
def _ensure_namespace_pkg(name: str, path: pathlib.PurePath) -> None:
    existing = sys.modules.get(name)
    if existing is not None and getattr(existing, "__path__", None):
        return
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(path)]
    sys.modules[name] = pkg


_ensure_namespace_pkg("huginn", _AGENT / "huginn")
_ensure_namespace_pkg("huginn.agent", _AGENT / "huginn" / "agent")

# import 时注册 13 个内置段 (与生产同一入口).
import huginn.agent.prompt_builder as _pb  # noqa: E402,F401
import huginn.plugins.prompt_segments as _ps  # noqa: E402

# 生产调用形状 (见 context._esp_prompt_segments / prompt_builder / autoloop.engine_act).
_PROMPT_CALLS: dict[str, tuple[object, ...]] = {
    "main": ("main", "chat", "execute", "s0_blank", None),
    "s7": ("research", "validate", "s7_self_modify", None),
    "autoloop": ("default", "execute", "", None),
}

# 冻结预算 (prompt + 工具定义字符数): 空 = observe 只测不判. 冻结后只许下调.
BUDGET: dict[str, int] = {}

# 容忍的已知失败 (owner, failure). 必须为空; 修复后必须摘除, 否则 fixed 侧报红.
KNOWN_FAILURES: list[tuple[str, str]] = []

# 模型面禁用标记: 大写占位标记 + 经典假文本. 小写 "todo" 是真实工具语义
# (todo_read_tool/todo_write_tool), 不在此列 —— 禁用词只锁"未写完的痕迹".
_PLACEHOLDER = re.compile(r"\b(?:TODO|FIXME|XXX|TBD)\b")
_FAKE_TEXT = re.compile(r"lorem ipsum", re.IGNORECASE)
_PLACEHOLDERS = (_PLACEHOLDER, _FAKE_TEXT)


def _render(path: str) -> str:
    if path == "main":
        return _ps.render_prompt_segments_for_path(*_PROMPT_CALLS[path])
    if path == "s7":
        return _ps.assemble_prompt_segments(*_PROMPT_CALLS[path])
    return _ps.assemble_plugin_prompt_segments(*_PROMPT_CALLS[path])


@pytest.fixture(scope="module")
def surface() -> tuple[dict[str, int], list[dict], list[tuple[str, str, str]]]:
    """(各路径 prompt 字符数, 工具 function 定义, 待检文本面)."""
    from huginn.tools import register_core_tools
    from huginn.tools.registry import ToolRegistry

    snapshot = ToolRegistry.snapshot()
    try:
        if not ToolRegistry.list_tools():
            register_core_tools()
        definitions = [dict(s["function"]) for s in ToolRegistry.get_all_schemas()]
    finally:
        # 全局注册表必须原样归还 (conftest._restore_tool_registry 卫生门).
        ToolRegistry.restore(snapshot)

    prompts = {path: _render(path) for path in _PROMPT_CALLS}
    surfaces: list[tuple[str, str, str]] = [
        (f"prompt/{path}", "huginn/agent/prompt_builder.py", text)
        for path, text in prompts.items()
    ]
    surfaces += [
        (
            f"tool/{d['name']}",
            "huginn/tools/",
            json.dumps(d, ensure_ascii=False, sort_keys=True),
        )
        for d in definitions
    ]
    return {path: len(text) for path, text in prompts.items()}, definitions, surfaces


def _failures(
    definitions: list[dict], surfaces: list[tuple[str, str, str]]
) -> dict[str, str]:
    """发现的问题 -> 归属文件 (与 dsh 的 Surface.owner 同语义)."""
    found: dict[str, str] = {}
    for name, owner, text in surfaces:
        if not text.strip():
            found[f"{owner}: {name}: 空文本"] = owner
        for pattern in _PLACEHOLDERS:
            for match in pattern.finditer(text):
                found[f"{owner}: {name}: 出现占位标记 {match.group(0)!r}"] = owner
    for d in definitions:
        if not (d.get("description") or "").strip():
            found[f"huginn/tools/: tool/{d.get('name')}: 描述为空"] = "huginn/tools/"
    return found


def _known() -> dict[str, str]:
    return {f"{owner}: {failure}": owner for owner, failure in KNOWN_FAILURES}


def test_model_surface_matches_known_failures(surface) -> None:
    """KNOWN_FAILURES 双向核对: 新漂移红, 已修复未摘除也红."""
    _prompts, definitions, surfaces = surface
    found = _failures(definitions, surfaces)
    known = _known()
    drift = sorted(set(found) - set(known))
    fixed = sorted(set(known) - set(found))
    assert not drift, "模型面出现未登记的失败:\n" + "\n".join(drift)
    assert not fixed, "已知失败已修复, 请从 KNOWN_FAILURES 摘除:\n" + "\n".join(fixed)


def test_reports_surface_measurements(surface) -> None:
    """observe 阶段: 输出实测字符数; 冻结预算后同一函数转为硬判."""
    prompts, definitions, _surfaces = surface
    tools = len(json.dumps(definitions, ensure_ascii=False))
    for path, chars in prompts.items():
        total = chars + tools
        print(f"[model-surface] {path}: prompt={chars} tools={tools} total={total}")
        if (limit := BUDGET.get(path)) is not None:
            assert total <= limit, (
                f"{path} 模型面 {total} 字符超出冻结预算 {limit}; "
                "预算只减不增, 上调须显式改 tests/test_model_surface.py"
            )
    assert all(chars > 0 for chars in prompts.values()), "prompt 段渲染为空"
    assert definitions, "工具注册面为空, 无法度量"
    print(f"[model-surface] tools={len(definitions)} (未冻结预算, observe only)")
