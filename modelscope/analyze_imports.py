"""Static import-dependency analyzer for Huginn tool seeds.

Resolves the transitive closure of huginn.* modules needed by each seed
tool module (using function-level/from imports), and aggregates the
third-party top-level packages imported anywhere in that closure.

Purely static — no module is executed. Heavy third-party libs may be absent.
"""
from __future__ import annotations

import ast
import os
from collections import defaultdict

ROOT = "/workspace/agent"
PKG = os.path.join(ROOT, "huginn")

SEEDS = [
    "huginn/tools/sci/unit_tool.py",
    "huginn/tools/design/doe_tool.py",
    "huginn/tools/design/design_atom_tool.py",
    "huginn/tools/design/gap_analysis_tool.py",
    "huginn/tools/sci/sklearn_tool.py",
    "huginn/tools/sci/numerical_tool.py",
    "huginn/tools/causal/__init__.py",
]


def huginn_to_path(mod: str) -> str | None:
    """Map a dotted 'huginn.a.b.c' module name to a source path under PKG."""
    assert mod.startswith("huginn")
    rel = mod.split(".")[1:]
    p = os.path.join(PKG, *rel)
    if os.path.isfile(p + ".py"):
        return p + ".py"
    if os.path.isfile(os.path.join(p, "__init__.py")):
        return os.path.join(p, "__init__.py")
    return None


def module_imports(path: str) -> list[tuple[str, str]]:
    """Return top-level import names as (top_package, module_dotted)."""
    try:
        tree = ast.parse(open(path, encoding="utf-8").read())
    except Exception as e:
        return [("__parse_error__", str(e))]
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out.append((a.name.split(".")[0], a.name))
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.append((node.module.split(".")[0], node.module))
    return out


def closure_for(seed: str) -> tuple[set[str], dict[str, list[str]], set[str]]:
    """Return (huginn_closure, ext_imports_by_mod, parse_errors)."""
    huginn_closure: set[str] = set()
    ext: dict[str, list[str]] = {}
    parse_errors: set[str] = set()
    queue = [(seed, seed)]

    seen_mods: set[str] = set()
    while queue:
        current, origin = queue.pop()
        path = os.path.join(ROOT, current)
        if not os.path.isfile(path):
            continue
        if current in seen_mods:
            continue
        seen_mods.add(current)
        huginn_closure.add(current)

        if current.endswith("__init__.py"):
            continue
        for top, dotted in module_imports(path):
            if top == "__parse_error__":
                parse_errors.add(current)
                continue
            if top == "huginn":
                p = huginn_to_path(dotted)
                rel = os.path.relpath(p, ROOT) if p else None
                if rel and rel not in queue and rel not in seen_mods:
                    queue.append((rel, current))
            else:
                ext.setdefault(current, [])
                if dotted not in ext[current]:
                    ext[current].append(dotted)

    return huginn_closure, ext, parse_errors


def main() -> None:
    ext_agg: dict[str, list[str]] = {}
    all_closed: set[str] = set()
    for seed in SEEDS:
        closure, ext, errs = closure_for(seed)
        all_closed |= closure
        for mod, tops in ext.items():
            ext_agg.setdefault(mod, [])
            for t in tops:
                if t not in ext_agg[mod]:
                    ext_agg[mod].append(t)
        print(f"\n==== {seed}  ({len(closure)} huginn modules) ====")
        if errs:
            print("  parse errors:", sorted(errs))
        for mod in sorted(closure):
            if mod in ext_agg:
                print("  " + mod, "->", ", ".join(ext_agg[mod]))

    # aggregated third-party across all seeds
    base = set()
    for mod, tops in ext_agg.items():
        base |= {t for t in tops if not t.startswith("__")}
    stdlib = {
        "abc", "argparse", "ast", "asyncio", "base64", "collections",
        "contextlib", "contextvars", "copy", "csv", "ctypes", "dataclasses",
        "datetime", "decimal", "enum", "functools", "hashlib", "http",
        "importlib", "inspect", "io", "itertools", "json", "logging", "math",
        "numbers", "operator", "os", "pathlib", "platform", "queue",
        "random", "re", "socket", "ssl", "statistics", "string", "sys", "textwrap",
        "time", "traceback", "types", "typing", "urllib", "uuid", "warnings",
        "webbrowser", "zipfile", "signal", "tempfile", "shutil", "subprocess",
        "struct", "weakref", "fractions", "bisect", "heapq", "tokenize",
        "dis", "codecs", "array", "sqlite3", "gzip", "marshal", "pickle",
        "multiprocessing", "concurrent", "threading", "tarfile", "glob",
        "__future__", "unicodedata", "difflib", "string", "binascii",
        "gettext", "locale", "resource", "pydoc", "running", "_thread", "select",
    }
    third = sorted(base - stdlib - {"__future__"})
    print("\n==== THIRD-PARTY (aggregated) ====")
    print(", ".join(third))
    print("\n==== huginn closure file count:", len(all_closed), "====")


if __name__ == "__main__":
    main()