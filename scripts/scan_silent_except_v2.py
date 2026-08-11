"""AST scanner v2: find TRULY silent except handlers.

A handler is "truly silent" if its body is 1-2 statements AND none of them:
- calls a logger method
- calls print()
- re-raises
- defines a function/class
- assigns to anything other than a simple sentinel (None/False)
- contains a non-trivial expression (function call, conditional, etc.)

Target bodies: {pass, return <const>, continue, break} optionally with one
simple `var = None` assignment.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

LOG_METHODS = {"debug", "info", "warning", "warn", "error", "exception", "critical", "log"}

# Allowed simple return constants (ast.NameConstant was removed in 3.14;
# ast.Constant subsumes it since 3.8)
SIMPLE_CONSTS = (ast.Constant,)


def _is_logging_call(node: ast.AST) -> bool:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr in LOG_METHODS:
            return True
    return False


def _is_print_call(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print"


def _has_reraise(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Raise) and child.exc is None:
            return True
    return False


def _body_is_truly_silent(body: list[ast.stmt]) -> bool:
    if not body:
        return False
    if len(body) > 2:
        return False
    for stmt in body:
        # pass
        if isinstance(stmt, ast.Pass):
            continue
        # return <const>
        if isinstance(stmt, ast.Return):
            if stmt.value is None:
                continue
            v = stmt.value
            if isinstance(v, SIMPLE_CONSTS):
                continue
            # return None/False/True via Name (py<3.8 leftover, rare)
            if isinstance(v, ast.Name) and v.id in {"None", "True", "False"}:
                continue
            # return [] / {} / () — empty containers
            if isinstance(v, (ast.List, ast.Tuple, ast.Dict, ast.Set)) and len(getattr(v, "elts", getattr(v, "keys", []))) == 0:
                continue
            return False
        # continue / break
        if isinstance(stmt, (ast.Continue, ast.Break)):
            continue
        # var = None / var = False / var = "" — simple sentinel assignment
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                return False
            if isinstance(stmt.value, SIMPLE_CONSTS):
                continue
            if isinstance(stmt.value, ast.Name) and stmt.value.id in {"None", "True", "False"}:
                continue
            return False
        # Any logging/print anywhere in body
        for child in ast.walk(stmt):
            if _is_logging_call(child) or _is_print_call(child):
                return False
        return False
    return True


def scan_file(path: Path) -> list[tuple[int, str]]:
    try:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return []
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        if _has_reraise(node):
            continue
        if not _body_is_truly_silent(node.body):
            continue
        # Also exclude if any logging/print deeper in body (defensive)
        if any(_is_logging_call(c) or _is_print_call(c) for c in ast.walk(node)):
            continue
        exc_type = ""
        if node.type:
            if isinstance(node.type, ast.Name):
                exc_type = node.type.id
            elif isinstance(node.type, ast.Tuple):
                exc_type = ", ".join(
                    e.id for e in node.type.elts if isinstance(e, ast.Name)
                )
            else:
                exc_type = "complex"
        hits.append((node.lineno, exc_type or "bare"))
    return hits


def main(roots: list[str]) -> None:
    total = 0
    per_file: list[tuple[Path, list[tuple[int, str]]]] = []
    for root in roots:
        for path in Path(root).rglob("*.py"):
            if "/.venv/" in str(path) or "/__pycache__/" in str(path):
                continue
            hits = scan_file(path)
            if hits:
                per_file.append((path, hits))
                total += len(hits)
    per_file.sort(key=lambda x: -len(x[1]))
    for path, hits in per_file[:40]:
        print(f"{len(hits):4d}  {path}")
    print(f"\nTotal TRULY silent handlers: {total} across {len(per_file)} files")


if __name__ == "__main__":
    main(sys.argv[1:] or ["huginn"])
