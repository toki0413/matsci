"""模块普查双向门 — 移植自 dsh ``tests/docs.spec.ts`` 的「登记表 ↔ 代码树」双向断言.

原则 (来自 dsh docs.spec):
- 枚举是**唯一事实源**: 代码树里出现的每个模块, 登记表必须有一行;
- 反向同样成立: 登记表里的每行必须对应真实模块 (无幽灵条目);
- 漏登记与多登记都会失败, 不允许"只查一边"。

为什么需要: ``research/huginn_module_census.csv`` 是模块保留/迁移/删除的唯一决策记录。
若新增模块可以不入表, 普查表就会逐渐退化成历史快照 —— 这正是 dsh 用测试而不是
靠自觉来守住的东西。孤儿模块 (``task_contract``/``trace``) 就是这样被发现的。

失败时怎么办:
- ``missing``: 在 CSV 里补一行 (module + 可达性 + 证据 + verdict);
- ``ghost``: 该模块已删除/改名, 删掉或更正对应行;
- verdict 非法: 只能用 KEEP / MOVE / SUSPECT / DELETE / ENTRY 五类.
"""

from __future__ import annotations

import csv
import pathlib
import re

_REPO = pathlib.Path(__file__).resolve().parents[1]
_CENSUS = _REPO.parent / "research" / "huginn_module_census.csv"
_HUGINN = _REPO / "huginn"

# dsh 的裁决是**封闭集合**: 出现第六种就说明有人在绕过分类而不是做决策.
_VERDICT_KINDS = ("KEEP", "MOVE", "SUSPECT", "DELETE", "ENTRY")
_VERDICT_RE = re.compile(r"^(%s)\(([^)]+)\)$" % "|".join(_VERDICT_KINDS))


def _tree_modules() -> set[str]:
    """枚举 huginn/ 的模块标识 (包目录取目录名, 模块取文件名 stem).

    - 目录: 递归含有 ``*.py`` 才算模块; 纯资产目录 (如 ``assets/`` 只有图片) 不算;
    - 顶层 ``__init__.py``: 是包本身, 不是可独立登记的模块, 排除;
    - ``__pycache__`` / 隐藏目录: 构建产物, 排除.
    """
    mods: set[str] = set()
    for p in _HUGINN.iterdir():
        if p.name.startswith(".") or p.name == "__pycache__":
            continue
        if p.is_dir():
            if any(p.rglob("*.py")):
                mods.add(p.name)
        elif p.suffix == ".py" and p.stem != "__init__":
            mods.add(p.stem)
    return mods


def _census_rows() -> list[dict[str, str]]:
    assert _CENSUS.is_file(), f"缺模块普查表: {_CENSUS}"
    with _CENSUS.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_census_covers_exactly_tree_modules():
    """双向覆盖: 树里每个模块都已登记, 登记表里没有幽灵行."""
    rows = _census_rows()
    registered = {r["module"].strip() for r in rows}
    tree = _tree_modules()
    missing = sorted(tree - registered)
    ghost = sorted(registered - tree)
    assert not missing and not ghost, (
        "模块普查表与代码树漂移 (双向门):\n"
        f"  漏登记 (missing): {missing}\n"
        f"  幽灵行 (ghost):   {ghost}\n"
        f"请更新 {_CENSUS} 后重跑。"
    )


def test_module_rows_are_unique():
    """一个模块只能有一行; 重复行会让覆盖断言失真."""
    modules = [r["module"].strip() for r in _census_rows()]
    dupes = sorted({m for m in modules if modules.count(m) > 1})
    assert not dupes, f"普查表存在重复模块行: {dupes}"


def test_verdicts_belong_to_closed_set():
    """裁决必须是五类之一, 且带非空理由 (不能写裸 ``KEEP``)."""
    bad = [
        f"{r['module']}: {r['verdict']!r}"
        for r in _census_rows()
        if not _VERDICT_RE.match(r["verdict"].strip())
    ]
    assert not bad, (
        f"非法裁决 (只允许 {_VERDICT_KINDS} 且必须带理由): {bad}"
    )


def test_keep_and_move_rows_cite_importer_evidence():
    """KEEP/MOVE 是"继续养/继续搬"的承诺, 必须有实际引用方作为证据."""
    unsupported = [
        r["module"]
        for r in _census_rows()
        if r["verdict"].strip().startswith(("KEEP", "MOVE"))
        and not r["importer_kinds"].strip()
    ]
    assert not unsupported, (
        f"以下模块判定 KEEP/MOVE 却没有 importer 证据: {unsupported}"
    )
