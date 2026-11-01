"""Catalog handle 契约门 —— kind / 采集器 / handle 派生三方闭合.

Critical 形态: `discover_all` 是覆盖式采集, 某个 kind 的 handle 没传 → 该 kind 返回
空列表并把追踪视图里的旧条目冲掉, 「结构上永不出现」且门全绿. 本门把
`KIND_HANDLE_SPECS` (kind → 所需 handle) 作为契约, 断言:

- 声明的 kind 集 == `catalog.models.KINDS`;
- 声明的 handle 集 == `discover_handles()` 实际产出的键集 (双向);
- `_kind_collectors` 源码里读取的 handle 键 == 声明的 handle 集;
- 误传未知 handle 显式报错 (不再静默忽略).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from huginn.catalog.manager import KIND_HANDLE_SPECS, CatalogManager, discover_handles
from huginn.catalog.models import KINDS

_AGENT = Path(__file__).resolve().parents[1]
_MANAGER = _AGENT / "huginn" / "catalog" / "manager.py"


class _Ctx:
    """最小 ServerContext 替身: 三个 handle 槽位全部存在 (值为 None 也算存在)."""

    mcp_manager = None
    agent_factory = None


def _declared_handles() -> set[str]:
    return {v for v in KIND_HANDLE_SPECS.values() if v}


def test_kind_handle_specs_cover_all_kinds() -> None:
    assert set(KIND_HANDLE_SPECS) == set(KINDS), (
        f"KIND_HANDLE_SPECS 与 KINDS 不一致: "
        f"缺 {sorted(set(KINDS) - set(KIND_HANDLE_SPECS))}, "
        f"多 {sorted(set(KIND_HANDLE_SPECS) - set(KINDS))}"
    )


def test_discover_handles_matches_declared_set() -> None:
    produced = set(discover_handles(_Ctx()))
    declared = _declared_handles()
    assert produced == declared, (
        f"discover_handles 产出 {sorted(produced)} 与 KIND_HANDLE_SPECS 声明 "
        f"{sorted(declared)} 不一致 —— 新增 kind 忘了加 handle = 结构上永不出现"
    )


def test_kind_collectors_read_only_declared_handles() -> None:
    tree = ast.parse(_MANAGER.read_text(encoding="utf-8"))
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_kind_collectors"
    )
    read: set[str] = set()
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            read.add(node.args[0].value)
    assert read == _declared_handles(), (
        f"_kind_collectors 读取的 handle {sorted(read)} 与声明 "
        f"{sorted(_declared_handles())} 不一致"
    )


def test_unknown_handle_is_rejected() -> None:
    with pytest.raises(TypeError, match="unknown catalog handle"):
        CatalogManager().discover_all(bogus=object())
