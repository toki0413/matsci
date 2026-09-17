"""工具层的重编排受控 — 顶层 import 不得直连运行时核心.

对齐 ApxInf 的 "backend crates never import model concepts" 精神: `huginn/tools/`
里的工具是给 agent 用的执行原语, 理论上应是相对叶子 — import 一个工具不应**连带
把你整个 autoloop / agents / research / metacog 在 import 阶段就拉进内存**.

本门锁住的是**"模块顶层(runtime) import"**, 两个自然豁免:
  - 函数体内的懒加载 import(用到才 `from huginn.autoloop... import`, 允许);
  - `if TYPE_CHECKING:` 块内的 import(纯类型解析, 运行时不执行, 允许).

允许的依赖: `validation` / `memory` / `rag` / `core_types` / `utils` 等轻量
数据/校验层(它们是 import 轻、不连带拖垮 runtim)。

被禁的: `autoloop` / `agents` / `research` / `metacog` / `execution` /
`evolution` / `exploration` —— 这些是重编排核心, 工具不得在顶层硬连。
"""
from __future__ import annotations

import ast
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_TOOLS = _ROOT / "huginn" / "tools"

# 工具不得在模块顶层 import 的重编排核心.
_FORBIDDEN = {
    "autoloop", "agents", "research", "metacog",
    "execution", "evolution", "exploration",
}


def _top_level_huginn_imports(path: Path) -> list[str]:
    """返回文件里**模块顶层**(直接父节点=Module)的 `huginn.*` import。

    直接父判断只认「import 直接挂在 Module 下」。函数体内的懒加载 import(父是
    FunctionDef)与 `if TYPE_CHECKING:`(父是 If)天然被排除 —— 它们不是运行时
    顶层import, 不破坏"import 一个工具不连带拉进运行时"的目标。
    """
    tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    parent_map: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parent_map[id(child)] = parent
    hits: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if not isinstance(parent_map.get(id(node)), ast.Module):
            continue  # 非顶层(函数内/TYPE_CHECKING)懒加载, 放行
        if isinstance(node, ast.Import):
            modules = [a.name for a in node.names]
        else:
            modules = [node.module] if node.module else []
        for m in modules:
            if m == "huginn" or m.startswith("huginn."):
                hits.append(m)
    return sorted(set(hits))


def _forbidden(hits: list[str]) -> list[str]:
    bad: list[str] = []
    for h in hits:
        parts = h.split(".")
        # parts[0] 恒为 "huginn"; 只有与顶层子包名匹配才算上探.
        if len(parts) >= 2 and parts[1] in _FORBIDDEN:
            bad.append(h)
    return sorted(bad)


def test_tools_no_top_level_runtime_core_import() -> None:
    offenders: list[str] = []
    for f in sorted(_TOOLS.rglob("*.py")):
        bad = _forbidden(_top_level_huginn_imports(f))
        if bad:
            offenders.append(f"{f.relative_to(_ROOT)}: {', '.join(bad)}")
    assert not offenders, (
        "huginn/tools 模块顶层直接 import 了重编排核心(autoloop/agents/research/"
        "metacog/execution/evolution/exploration)。这会让 import 一个工具连带把整个"
        "agent 运行时拉进内存, 破坏工具层独立性。请改为函数体内懒加载, 或把确实"
        "仅用于类型注解的 import 收进 `if TYPE_CHECKING:`。违规:\n"
        + "\n".join(offenders)
    )
