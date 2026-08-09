#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
huginn 核心库 print() 技术债务审计脚本。

扫描 /workspace/agent/huginn 下所有 .py 文件，找出 print( 调用并按上下文分类:
  A. selfcheck/demo        — __main__ 块 / _selfcheck 等演示函数 / selfcheck-demo-test 命名文件 / 顶部 docstring 自检标注
  B. plugins/third-party   — plugins/science-skills/ 下的外部 skill 脚本
  C. 科学实验/可视化        — metacog 中明显数学/拓扑实验模块里的 print
  D. 核心 lib 逻辑 (需修复) — 核心库业务逻辑里用 print 做日志、本应改用 logger 的地方

只读扫描，不修改任何源码。使用 ast 精确定位 __main__ 块与函数范围。
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path("/workspace/agent/huginn")
REPORT = Path("/workspace/_print_audit_report.txt")

# ── C 类: 明显是数学/拓扑实验模块 (任务列出的 7 个 + "等" 同类拓扑/数学模块) ──
EXPERIMENT_MODULES: set[str] = {
    # 任务显式列出
    "metacog/composite_token_experiment.py",
    "metacog/structure_cognitive_map.py",
    "metacog/persistence_landscape.py",
    "metacog/category_functor.py",
    "metacog/simplicial_homology.py",
    "metacog/topology_protocol.py",
    "metacog/sheaf_cohomology.py",
    # "等" — 同属拓扑/数学实验性质的模块
    "metacog/topology_lens.py",
    "metacog/trace_topology.py",
    "metacog/unified_complex.py",
    "metacog/hypothesis_manifold.py",
}

# 可视化函数名 (用于实验模块内 C 类判定)
VIZ_FUNC_RE = re.compile(r"(visualiz|plot|render|draw|display|show|demo|print_)", re.IGNORECASE)


def _is_demo_func_name(name: str) -> bool:
    """判断函数名是否为 演示/自检 函数 (前缀/后缀/任意位置含 selfcheck, 或精确 main/demo)."""
    if re.search(r"self_?check", name, re.IGNORECASE):
        return True
    if re.match(r"^(main|_main|demo|_demo)$", name):
        return True
    if re.match(r"^(demo_|_demo_)", name):
        return True
    return False

# selfcheck / demo / test 命名文件 (整个文件 → A 类)
SELFCHK_FILE_RE = re.compile(r"(_self_?check|_demo|_test_|_test\.py$|^test_)", re.IGNORECASE)

# 顶部 docstring 自检标注关键词
SELFCHECK_DOC_KEYWORDS = ("self-check", "self check", "selfcheck", "自检", "自测", "示范", "demo")

# 注释行里的 print (正则 fallback 用)
PRINT_LINE_RE = re.compile(r"\bprint\s*\(")


# ─────────────────────────── AST 辅助 ───────────────────────────

def _test_has_main_guard(test) -> bool:
    """test 节点是否含 `__name__ == "__main__"` 比较 (含 `X and Y` 这类 BoolOp)."""
    if isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq):
        left, right = test.left, (test.comparators[0] if test.comparators else None)
        for a, b in ((left, right), (right, left)):
            if (
                isinstance(a, ast.Name) and a.id == "__name__"
                and isinstance(b, ast.Constant) and b.value == "__main__"
            ):
                return True
        return False
    if isinstance(test, ast.BoolOp):
        return any(_test_has_main_guard(v) for v in test.values)
    return False


def _is_main_guard(node: ast.If) -> bool:
    """判断 if 节点是否为 `if __name__ == "__main__":` (含 `and ...` 复合守卫) 守卫。"""
    return isinstance(node, ast.If) and _test_has_main_guard(node.test)


def _bare_call_names(nodes: Iterable[ast.stmt]) -> set[str]:
    """收集一组语句里所有 `foo()` 形式 (bare Name 调用) 的函数名。"""
    names: set[str] = set()
    for n in nodes:
        for sub in ast.walk(n):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                names.add(sub.func.id)
    return names


def _module_docstring_text(tree) -> str:
    if tree is None:
        return ""
    try:
        ds = ast.get_docstring(tree)
    except Exception:
        return ""
    return (ds or "").lower()


# ─────────────────────────── 文件级分析 ───────────────────────────

class FileAnalysis:
    def __init__(self, rel: str, src: str, tree: ast.Module):
        self.rel = rel
        self.src = src
        self.tree = tree
        self.lines = src.splitlines()
        # 所有函数定义 (start, end, name)
        self.funcs: list[tuple[int, int, str]] = []
        # __main__ 块范围 (start, end)
        self.main_ranges: list[tuple[int, int]] = []
        # 被 __main__ 直接调用的函数名
        self.main_called: set[str] = set()
        # 演示函数范围 (start, end, name) — __main__ 调用的 / 名字匹配 demo 模式
        self.demo_func_ranges: list[tuple[int, int, str]] = []
        self._analyze()

    def _analyze(self) -> None:
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.funcs.append((node.lineno, node.end_lineno or node.lineno, node.name))
            elif isinstance(node, ast.If) and _is_main_guard(node):
                start = node.lineno
                end = node.end_lineno or node.lineno
                self.main_ranges.append((start, end))
                self.main_called |= _bare_call_names(node.body)

        # demo 函数: 被 __main__ 调用, 或名字匹配 selfcheck/demo 模式
        for start, end, name in self.funcs:
            if name in self.main_called or _is_demo_func_name(name):
                self.demo_func_ranges.append((start, end, name))

    def enclosing_funcs(self, line: int) -> list[str]:
        return [name for (s, e, name) in self.funcs if s <= line <= e]

    def in_main_block(self, line: int) -> bool:
        return any(s <= line <= e for (s, e) in self.main_ranges)

    def in_demo_func(self, line: int) -> bool:
        return any(s <= line <= e for (s, e, _) in self.demo_func_ranges)

    def in_viz_func(self, line: int) -> bool:
        return any(
            s <= line <= e and VIZ_FUNC_RE.search(name)
            for (s, e, name) in self.funcs
        )


# ─────────────────────────── print 收集 ───────────────────────────

def find_prints(fa: FileAnalysis) -> list[tuple[int, str]]:
    """返回 (行号, 该行源码) 列表。优先用 ast, 语法错误时退回正则。"""
    out: list[tuple[int, str]] = []
    try:
        for node in ast.walk(fa.tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
                ln = node.lineno
                line = fa.lines[ln - 1] if 0 < ln <= len(fa.lines) else ""
                out.append((ln, line))
        return out
    except Exception:
        pass
    # fallback: 正则 + 注释剔除
    for i, raw in enumerate(fa.lines, start=1):
        code = raw.split("#", 1)[0]
        if PRINT_LINE_RE.search(code):
            out.append((i, raw))
    return out


# ─────────────────────────── 分类 ───────────────────────────

def classify(fa: FileAnalysis, line: int, rel: str) -> str:
    # B: 第三方插件
    if rel.startswith("plugins/science-skills/"):
        return "B"

    is_experiment = rel in EXPERIMENT_MODULES

    # C: 实验模块 — 整体视为实验代码 (非核心 lib 债务)
    if is_experiment:
        return "C"

    # A: selfcheck/demo 文件名 / 顶部 docstring 自检标注 → 整文件 A
    if SELFCHK_FILE_RE.search(Path(rel).name):
        return "A"
    if any(kw in _module_docstring_text(fa.tree) for kw in SELFCHECK_DOC_KEYWORDS):
        return "A"

    # A: 在 __main__ 块 或 演示函数里
    if fa.in_main_block(line) or fa.in_demo_func(line):
        return "A"

    # 否则: 核心 lib 业务逻辑 print → D
    return "D"


# ─────────────────────────── 主流程 ───────────────────────────

def iter_py_files() -> list[Path]:
    files = []
    for p in ROOT.rglob("*.py"):
        # 跳过 __pycache__
        if "__pycache__" in p.parts:
            continue
        files.append(p)
    return sorted(files)


def main() -> None:
    files = iter_py_files()

    records: list[tuple[str, str, int, str]] = []  # (cat, rel, line, code)
    parse_errors: list[str] = []

    for p in files:
        try:
            src = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = str(p.relative_to(ROOT))
        tree = None
        try:
            tree = ast.parse(src, filename=str(p))
        except SyntaxError:
            parse_errors.append(rel)
            # 用正则 fallback 仍要尝试收集 print
            fa = FileAnalysis.__new__(FileAnalysis)
            fa.rel, fa.src, fa.tree, fa.lines = rel, src, None, src.splitlines()
            fa.funcs, fa.main_ranges, fa.main_called, fa.demo_func_ranges = [], [], set(), []
            # docstring 取不到, 直接置空
            pass
        if tree is not None:
            fa = FileAnalysis(rel, src, tree)
        for ln, code in find_prints(fa):
            cat = classify(fa, ln, rel)
            records.append((cat, rel, ln, code.strip()))

    # 统计
    from collections import defaultdict
    by_cat = defaultdict(int)
    by_cat_files = defaultdict(lambda: defaultdict(int))
    for cat, rel, ln, code in records:
        by_cat[cat] += 1
        by_cat_files[cat][rel] += 1

    total = len(records)

    # 组装报告
    lines: list[str] = []
    w = lines.append
    w("=" * 70)
    w("huginn 核心库 print() 技术债务审计报告")
    w("=" * 70)
    w(f"扫描根目录          : {ROOT}")
    w(f"扫描 .py 文件数     : {len(files)}")
    w(f"语法解析失败(正则兜底): {len(parse_errors)}")
    if parse_errors:
        w(f"  -> {', '.join(parse_errors[:10])}{' ...' if len(parse_errors) > 10 else ''}")
    w(f"总 print( 调用数    : {total}")
    w("")
    w("─ 分类分布 ────────────────────────────────────────────────")
    w(f"  A. selfcheck/demo          : {by_cat['A']}")
    w(f"  B. plugins/third-party     : {by_cat['B']}")
    w(f"  C. 科学实验/可视化          : {by_cat['C']}")
    w(f"  D. 核心 lib 逻辑 (真正需修复): {by_cat['D']}   ← 需处理")
    w("")
    w("=" * 70)
    w(f"D 类完整列表 ({by_cat['D']} 项) — 真正需要修复的核心库 print")
    w("=" * 70)
    d_items = sorted([(rel, ln, code) for (cat, rel, ln, code) in records if cat == "D"])
    if d_items:
        for rel, ln, code in d_items:
            w(f"{rel}:{ln}: {code}")
    else:
        w("(无)")
    w("")
    w("=" * 70)
    w("各类文件分布明细")
    w("=" * 70)
    for cat in ("A", "B", "C", "D"):
        w(f"\n[{cat}] 类 ({by_cat[cat]} 项):")
        items = sorted(by_cat_files[cat].items(), key=lambda kv: (-kv[1], kv[0]))
        for rel, c in items:
            w(f"  {c:4d}  {rel}")

    w("")
    w("─ 说明 ────────────────────────────────────────────────────")
    w("* print( 通过 ast 定位 (Call 节点 func.id == 'print'), 自动排除注释中的 # print(...).")
    w("* __main__ 块与演示函数 (_selfcheck / run_selfcheck / 被 __main__ 直接调用的函数等) 经 ast 范围匹配判为 A/C 类.")
    w("* C 类实验模块集合 (任务列出 7 个 + 同类拓扑/数学模块):")
    for m in sorted(EXPERIMENT_MODULES):
        w(f"    - {m}")
    w("* D 类 = 核心库业务逻辑里非 selfcheck/非实验/非第三方的 print, 视为应改用 logger 的技术债务.")

    report_text = "\n".join(lines)
    REPORT.write_text(report_text + "\n", encoding="utf-8")

    # 同时输出到 stdout
    print(report_text)


if __name__ == "__main__":
    main()
