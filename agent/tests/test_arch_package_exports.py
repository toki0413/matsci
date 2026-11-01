"""包导出面门 —— `__all__` 与模块公共名双向一致.

`__all__` 是包的公开契约. 两种静默漂移都无信号:
- `__all__` 写了但模块里没有 → `from pkg import *` 才炸, 平时全绿;
- 模块新增公共名却没进 `__all__` → 导出面悄悄扩张, 调用方开始依赖"非契约"名字.

本门用 weightless AST 扫 `huginn/**/__init__.py`:
- `__all__` 里的常量名必须能静态解析 (顶层定义 / 导入 / 子模块名);
  PEP 562 惰性导出的包 (定义 `__getattr__`) 豁免此项 —— 名字在运行时才解析.
- 顶层公共 def/class (非 `_` 开头, 非 logger) 必须出现在 `__all__`.
"""

from __future__ import annotations

import ast
from pathlib import Path

_HUGINN = Path(__file__).resolve().parents[1] / "huginn"


def _flatten_all(node: ast.AST) -> tuple[set[str], bool]:
    """展开 `__all__` 字面量; 返回 (常量名集合, 是否含动态片段)."""
    names: set[str] = set()
    dynamic = False

    def walk(value: ast.AST) -> None:
        nonlocal dynamic
        if isinstance(value, (ast.List, ast.Tuple)):
            for elt in value.elts:
                walk(elt)
        elif isinstance(value, ast.Constant) and isinstance(value.value, str):
            names.add(value.value)
        else:
            # `*_extra` / 变量引用 / 拼接: 静态不可知, 交由 __getattr__ 豁免逻辑.
            dynamic = True

    walk(node)
    return names, dynamic


def _is_logger_assign(node: ast.Assign | ast.AnnAssign) -> bool:
    value = node.value
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == "getLogger"
    )


def analyze_package(path: Path) -> dict[str, object]:
    """分析单个 `__init__.py`, 返回公共名 / `__all__` / 失配清单."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    all_node: ast.AST | None = None
    defined: set[str] = set()
    imported: set[str] = set()
    has_getattr = False

    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            all_node = node.value
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
            if node.name == "__getattr__":
                has_getattr = True
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            if _is_logger_assign(node):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    defined.add(target.id)
        elif isinstance(node, ast.ImportFrom):
            imported |= {a.asname or a.name for a in node.names}
        elif isinstance(node, ast.Import):
            imported |= {(a.asname or a.name).split(".")[0] for a in node.names}

    if all_node is None:
        return {
            "path": str(path),
            "has_all": False,
            "missing": [],
            "public_unexported": [],
        }

    declared, dynamic = _flatten_all(all_node)
    submodules = {
        child.stem
        for child in path.parent.iterdir()
        if child.is_dir() and not child.name.startswith("_")
    }
    resolvable = defined | imported | submodules | {"__version__"}
    # PEP 562 或动态片段: 运行时才解析, 静态判不了, 不误报.
    missing = [] if (has_getattr or dynamic) else sorted(declared - resolvable)
    public_unexported = sorted(
        name
        for name in defined
        if not name.startswith("_") and name not in declared
    )
    return {
        "path": str(path),
        "has_all": True,
        "missing": missing,
        "public_unexported": public_unexported,
    }


def scan_packages(root: Path) -> list[dict[str, object]]:
    return [analyze_package(p) for p in sorted(root.rglob("__init__.py"))]


def test_all_exports_resolve_in_package():
    """不变量: `__all__` 里的每个名字都能在包内静态解析 (惰性导出包豁免)."""
    bad = [
        (row["path"], row["missing"])
        for row in scan_packages(_HUGINN)
        if row["missing"]
    ]
    assert not bad, (
        f"以下包的 `__all__` 指向不存在的名字 (import * 时才炸): {bad}"
    )


def test_public_names_are_declared_in_all():
    """不变量: 顶层公共 def/class 必须进 `__all__` (导出面不得悄悄扩张)."""
    bad = [
        (row["path"], row["public_unexported"])
        for row in scan_packages(_HUGINN)
        if row["public_unexported"]
    ]
    assert not bad, (
        f"以下包有未导出公共名 (新增公开 API 未登记): {bad} —— 要么加进 __all__, "
        f"要么改名以 _ 开头 (声明它是内部实现)."
    )


def test_missing_export_detected_on_synthetic_tree(tmp_path: Path):
    """合成树验证: `__all__` 指向幽灵名 → 抓到."""
    pkg = tmp_path / "demo"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(
        'def real():\n    pass\n\n__all__ = ["real", "ghost"]\n', encoding="utf-8"
    )
    row = analyze_package(pkg / "__init__.py")
    assert row["missing"] == ["ghost"]


def test_undeclared_public_name_detected_on_synthetic_tree(tmp_path: Path):
    """合成树验证: 公共 def 未进 `__all__` → 抓到."""
    pkg = tmp_path / "demo"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(
        'def exported():\n    pass\n\ndef hidden_api():\n    pass\n\n__all__ = ["exported"]\n',
        encoding="utf-8",
    )
    row = analyze_package(pkg / "__init__.py")
    assert row["public_unexported"] == ["hidden_api"]


def test_lazy_getattr_package_is_exempt_on_synthetic_tree(tmp_path: Path):
    """合成树验证: PEP 562 惰性导出包不误报 (名字运行时才解析)."""
    pkg = tmp_path / "demo"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(
        'def __getattr__(name):\n    raise AttributeError(name)\n\n'
        '__all__ = ["lazy_name"]\n',
        encoding="utf-8",
    )
    row = analyze_package(pkg / "__init__.py")
    assert row["missing"] == []
