"""数据层(rag/memory)的依赖方向 — 不得上探运行时编排层.

对齐 ApxInf 的依赖单向原则, 但落在其反向: 检索数据层 `rag/` 与记忆数据层
`memory/` 是相对独立的"数据/持久化"层, 不应在模块顶层反向依赖运行时的编排核心
(`tools` / `autoloop` / `agents` / `research` / `metacog` / `execution` /
`evolution` / `exploration`) —— 否则一套检索/记忆能力就绑到了具体 agent 运行时,
无法复用。

放行项:
  - `rag/*_tool.py` 是工具适配器(bridge), 允许依赖 `huginn.tools.base`;
  - `memory` 依赖 `rag` (或反之) 是**数据层间平级依赖**, 不算上探, 放行;
  - 函数体内懒加载 与 `if TYPE_CHECKING:` 的 import 均非运行时顶层依赖, 放行。
"""
from __future__ import annotations

import ast
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_HUGINN = _ROOT / "huginn"

# 数据层不得上探的运行时编排核心.
_FORBIDDEN = {
    "tools", "autoloop", "agents", "research", "metacog",
    "execution", "evolution", "exploration",
}


def _top_level_huginn_imports(path: pathlib.Path) -> list[str]:
    """返回模块顶层(direct parent=Module)的 `huginn.*` import."""
    tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    parent_map: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parent_map[id(child)] = parent
    hits: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Import | ast.ImportFrom):
            continue
        if not isinstance(parent_map.get(id(node)), ast.Module):
            continue
        if isinstance(node, ast.Import):
            modules = [a.name for a in node.names]
        else:
            modules = [node.module] if node.module else []
        for m in modules:
            if m == "huginn" or m.startswith("huginn."):
                hits.append(m)
    return sorted(set(hits))


def _scan(module: str) -> list[str]:
    offenders: list[str] = []
    for f in sorted((_HUGINN / module).glob("*.py")):
        if module == "rag" and f.name.endswith("_tool.py"):
            continue  # 工具适配器 bridge, 允许依赖 tools.base
        bad = [m for m in _top_level_huginn_imports(f)
               if m.startswith("huginn.") and m.split(".")[1] in _FORBIDDEN]
        if bad:
            offenders.append(f"{f.relative_to(_ROOT)}: {', '.join(sorted(bad))}")
    return offenders


def test_rag_data_layer_not_reaching_into_runtime() -> None:
    offenders = _scan("rag")
    assert not offenders, (
        "huginn/rag 数据层在模块顶层 import 了运行时编排核心(tools/autoloop/"
        "agents/research/metacog/execution/evolution/exploration)。这会让检索层"
        "绑到具体 agent 运行时。工具适配器(`*_tool.py`)是 bridge 除外。违规:\n"
        + "\n".join(offenders) or "(空)")


def test_memory_data_layer_not_reaching_into_runtime() -> None:
    offenders = _scan("memory")
    assert not offenders, (
        "huginn/memory 数据层在模块顶层 import 了运行时编排核心(tools/autoloop/"
        "agents/research/metacog/execution/evolution/exploration)。记忆层应保持"
        "可复用的数据独立性。违规:\n"
        + "\n".join(offenders) or "(空)")
