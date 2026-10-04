"""全局导入环棘轮 (only-shrink) —— 锁住"纠缠度不许增长".

**实测 (绝对 + 相对 import 都计入)**: ``huginn`` 1040 个模块里, **181 个处在导入环
中**, 分 14 个强连通分量 —— 一个 134 个模块的核心 blob, 外加 13 个小环
(image_analysis / image_design / security / exploration / skills / neb ...).

这是真正的"为什么一切依赖一切"的结构来源: 分层目前只有 6 个成对 seam 锁
(``test_arch_seam_integrity`` / ``test_arch_rag_memory_seam`` / ``test_arch_tools_direction``
等), 它们只钉住了极少数边, 整图其余部分可以自由成环, 无人拦.

**不能一次性拆掉**: 134 个模块的 blob 是长期演化结果, 硬拆是高风险大重构. 因此
本测试与仓库既有 idiom 一致, 做**只减不增的棘轮**:

* 环内模块数 ``<= CYCLE_MODULE_BUDGET`` (当前 181);
* 环的分量数 ``<= CYCLE_COMPONENT_BUDGET`` (当前 14).

新增任何能把新模块卷进环的 import → 测试红. 拆环后请下调下面两个常量.
零运行时改动; 现行为已冻结, 只许收敛.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_HUGINN = REPO_ROOT / "agent" / "huginn"

# 只减不增的棘轮预算 (实测于当前 HEAD). 拆环后下调.
CYCLE_MODULE_BUDGET = 181
CYCLE_COMPONENT_BUDGET = 14


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
    """模块级 import 图的边 (只保留指向本包内已知模块的 import)."""
    known = set(mods)

    def pkg_of(mod: str) -> str:
        # 相对 import 的基准: 包自身 (有子模块) 或父包.
        if any(k.startswith(mod + ".") for k in known):
            return mod
        return mod.rsplit(".", 1)[0] if "." in mod else mod

    edges: dict[str, set[str]] = {m: set() for m in mods}
    for mod, path in mods.items():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            targets: list[str] = []
            if isinstance(node, ast.Import):
                targets = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:  # 相对 import: from .x import y / from ..x import z
                    up = pkg_of(mod).split(".")
                    if node.level > 1:
                        up = up[: len(up) - (node.level - 1)]
                    base = ".".join(up)
                    if node.module:
                        base = f"{base}.{node.module}" if base else node.module
                    if base:
                        targets = [base, *[f"{base}.{a.name}" for a in node.names]]
                elif node.module:
                    targets = [node.module, *[f"{node.module}.{a.name}" for a in node.names]]
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
        f"导入环内的模块数从预算 {CYCLE_MODULE_BUDGET} 涨到 {in_cycle} —— 纠缠度在增长. "
        "新 import 把更多模块卷进了环. 把共享部分下沉到不依赖业务层的基建模块, 或改用"
        f"延迟 import. 当前环:\n{detail}"
    )
    assert len(cycles) <= CYCLE_COMPONENT_BUDGET, (
        f"导入环分量数从预算 {CYCLE_COMPONENT_BUDGET} 涨到 {len(cycles)} —— 新增了独立环. "
        f"当前环:\n{detail}"
    )


def test_detector_self_test(tmp_path: Path):
    """门禁自检: 确认检测器能抓到已知环, 且不误报无环图 (防门禁自己失效)."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "a.py").write_text("from . import b\n", encoding="utf-8")
    (pkg / "b.py").write_text("from . import a\n", encoding="utf-8")
    assert len(_cycles(_edges(_modules(pkg, "pkg"), "pkg"))) == 1, "应抓到 a<->b 环"

    (pkg / "b.py").write_text("", encoding="utf-8")  # 断环
    assert not _cycles(_edges(_modules(pkg, "pkg"), "pkg")), "断环后不应再报环"