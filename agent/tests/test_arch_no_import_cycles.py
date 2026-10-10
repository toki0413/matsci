"""全局导入环棘轮 (only-shrink) —— 锁住"纠缠度不许增长".

**口径**: 只测**模块 import 时真正会执行**的 import 组成的环 (即"import 会炸 /
需要固定 import 顺序"的真环). 延迟 import (函数体内) 与 ``if TYPE_CHECKING:``
块永不参与 import-time 环, 故不计入 —— 它们本就是断环手段.

**实测**: ``huginn`` 1040 个模块里, import-time 环内模块 **0 个** (0 个分量).
(早期版本用 ``ast.walk`` 把延迟 import 也算边, 虚报 181 模块 / 14 分量; 其中
``metacog`` 的 4 模块真环已通过把 ``recall_audit_context`` 下沉到叶子模块
``huginn/metacog/audit_context.py`` 拆掉.)

这是"为什么一切依赖一切"的结构来源: 分层目前只有 6 个成对 seam 锁
(``test_arch_seam_integrity`` / ``test_arch_rag_memory_seam`` / ``test_arch_tools_direction``
等), 它们只钉住了极少数边, 整图其余部分可以自由成环, 无人拦.

**预算为零**: 任何新增的 import-time 环都会变红. 断环手段: 把共享部分下沉到不反向
依赖业务层的叶子模块, 或改用延迟 import / ``TYPE_CHECKING`` (这三者都能让本门禁的
计数真正下降 —— 口径与药方一致).
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_HUGINN = REPO_ROOT / "agent" / "huginn"

# 只减不增的棘轮预算: import-time 真环已清零, 锁死在此 (不得上调).
CYCLE_MODULE_BUDGET = 0
CYCLE_COMPONENT_BUDGET = 0


def _modules(root: Path, prefix: str) -> dict[str, Path]:
    """收集 ``root`` 下所有模块: ``{点分模块名: 文件路径}``."""
    mods: dict[str, Path] = {}
    for p in root.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        parts = list(p.relative_to(root).with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        mods[".".join([prefix, *parts])] = p
    return mods


def _edges(mods: dict[str, Path], prefix: str) -> dict[str, set[str]]:
    """模块级 **import-time** 依赖图的边 (只保留指向本包内已知模块的 import).

    只计入模块 import 时**真正会执行**的 import statement:
      * ``if TYPE_CHECKING:`` 块内的 import —— 运行时不执行, 不算;
      * 函数/方法体内的 import —— 延迟 import 正是断环的手段, 不算;
      * class 体 / ``try`` 块内的 import —— import 时执行, 计入.

    早期版本用 ``ast.walk`` 把函数体内的延迟 import 也算作边, 于是把"用延迟
    import 断开的环"重新计入 (自相矛盾: 它开出的药方自己不吃), 把纠缠度虚高
    ~45 倍 (181/14 → 真实 0). 现在只测"import 时会炸/需要固定顺序"的真环.
    """
    known = set(mods)

    def pkg_of(mod: str) -> str:
        # 相对 import 的基准: 包自身 (有子模块) 或父包.
        if any(k.startswith(mod + ".") for k in known):
            return mod
        return mod.rsplit(".", 1)[0] if "." in mod else mod

    def base_of(mod: str, level: int, module: str | None) -> str:
        up = pkg_of(mod).split(".")
        if level > 1:
            up = up[: len(up) - (level - 1)]
        base = ".".join(up)
        if module:
            base = f"{base}.{module}" if base else module
        return base

    def collect(node: ast.AST, out: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
                continue  # 延迟 import: 断环手段, 不算边
            if isinstance(child, ast.If) and "TYPE_CHECKING" in ast.dump(child.test):
                continue  # 永不执行
            if isinstance(child, ast.Import):
                out.extend(a.name for a in child.names)
            elif isinstance(child, ast.ImportFrom):
                if child.level:
                    b = base_of(mod, child.level, child.module)
                    if b:
                        out.extend([b, *[f"{b}.{a.name}" for a in child.names]])
                elif child.module:
                    out.extend(
                        [child.module, *[f"{child.module}.{a.name}" for a in child.names]]
                    )
            collect(child, out)

    edges: dict[str, set[str]] = {m: set() for m in mods}
    for mod, path in mods.items():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except (SyntaxError, ValueError):
            continue
        targets: list[str] = []
        collect(tree, targets)
        for t in targets:
            if t.startswith(prefix) and t in known and t != mod:
                edges[mod].add(t)
    return edges


def _cycles(edges: dict[str, set[str]]) -> list[list[str]]:
    """Tarjan: 返回所有大小 >1 的强连通分量 (即导入环), 按规模降序."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    counter = [0]
    sccs: list[list[str]] = []

    def strong(v: str) -> None:
        index[v] = low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on_stack.add(v)
        for w in edges.get(v, ()):
            if w not in index:
                strong(w)
                low[v] = min(low[v], low[w])
            elif w in on_stack:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp: list[str] = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                comp.append(w)
                if w == v:
                    break
            sccs.append(comp)

    import sys

    prev = sys.getrecursionlimit()
    sys.setrecursionlimit(max(prev, len(edges) * 2 + 1000))
    try:
        for v in sorted(edges):
            if v not in index:
                strong(v)
    finally:
        sys.setrecursionlimit(prev)
    return sorted((c for c in sccs if len(c) > 1), key=len, reverse=True)


def test_import_cycle_budget_is_not_exceeded():
    """不变量: 环内模块数与环的分量数只减不增 (纠缠度棘轮)."""
    mods = _modules(_HUGINN, "huginn")
    assert len(mods) > 900, f"扫描到的模块数异常偏少 ({len(mods)}), 门禁可能失效"
    cycles = _cycles(_edges(mods, "huginn"))
    in_cycle = len({m for c in cycles for m in c})
    detail = "\n".join(
        f"  [{len(c)}] {', '.join(sorted(c)[:8])}{' ...' if len(c) > 8 else ''}"
        for c in cycles[:12]
    )
    assert in_cycle <= CYCLE_MODULE_BUDGET, (
        f"import-time 环内模块数从预算 {CYCLE_MODULE_BUDGET} 涨到 {in_cycle} —— "
        "新 import 把模块卷进了环. 断环: 把共享部分下沉到不依赖业务层的叶子模块, "
        f"或改用延迟 import / TYPE_CHECKING. 当前环:\n{detail}"
    )
    assert len(cycles) <= CYCLE_COMPONENT_BUDGET, (
        f"import-time 环分量数从预算 {CYCLE_COMPONENT_BUDGET} 涨到 {len(cycles)} —— "
        f"新增了独立环. 当前环:\n{detail}"
    )


def test_detector_self_test(tmp_path: Path):
    """门禁自检: 抓真环, 且**不误报**被延迟/类型 import 断开的环 (防口径回退)."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "a.py").write_text("from . import b\n", encoding="utf-8")
    (pkg / "b.py").write_text("from . import a\n", encoding="utf-8")
    assert len(_cycles(_edges(_modules(pkg, "pkg"), "pkg"))) == 1, "应抓到 a<->b 真环"

    # 函数体延迟 import 断环 —— 不得再报环.
    (pkg / "b.py").write_text(
        "def f():\n    from . import a\n    return a\n", encoding="utf-8"
    )
    assert not _cycles(_edges(_modules(pkg, "pkg"), "pkg")), "延迟 import 应断环"

    # TYPE_CHECKING 块断环 —— 同样不得报环.
    (pkg / "b.py").write_text(
        "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from . import a\n",
        encoding="utf-8",
    )
    assert not _cycles(_edges(_modules(pkg, "pkg"), "pkg")), "TYPE_CHECKING 应断环"

    # 彻底删除 import —— 不应再报环.
    (pkg / "b.py").write_text("", encoding="utf-8")
    assert not _cycles(_edges(_modules(pkg, "pkg"), "pkg")), "断环后不应再报环"
