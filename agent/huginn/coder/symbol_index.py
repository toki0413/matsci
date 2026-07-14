"""Lightweight symbol index for cross-file refactoring.

Extracts definitions (classes, functions, methods) and references from source
code. Python is supported natively via ``ast``; other languages fall back to
simple regex heuristics so the refactor engine still has something to work with.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Symbol:
    """A named symbol defined in the codebase."""

    name: str
    kind: str  # "class", "function", "method", "variable"
    file: str
    line: int
    col: int = 0


@dataclass
class Reference:
    """A usage of a symbol name in a file."""

    name: str
    file: str
    line: int
    col: int = 0


@dataclass
class IndexedFile:
    """Symbols and references extracted from a single file."""

    path: Path
    symbols: list[Symbol] = field(default_factory=list)
    references: list[Reference] = field(default_factory=list)


class SymbolIndex:
    """Index symbols and references across a workspace."""

    # File extensions that we know how to index.
    EXTENSIONS = {".py", ".js", ".ts", ".jsx", ".tsx", ".rs", ".go", ".java"}

    SKIP_DIRS = {
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        "node_modules",
        "target",
        "dist",
        "build",
        ".huginn_kb",
        ".chroma",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".kimi",
        ".kimi-code",
    }

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self._symbols: dict[str, list[Symbol]] = {}
        self._references: dict[str, list[Reference]] = {}
        self._files: list[IndexedFile] = []

    def build(self, extensions: set[str] | None = None) -> None:
        """Index all matching files under the workspace root."""
        extensions = extensions or self.EXTENSIONS
        self._symbols.clear()
        self._references.clear()
        self._files = []

        for ext in extensions:
            for path in self.root.rglob(f"*{ext}"):
                if self._should_skip(path):
                    continue
                indexed = self._index_file(path)
                if indexed:
                    self._files.append(indexed)
                    for sym in indexed.symbols:
                        self._symbols.setdefault(sym.name, []).append(sym)
                    for ref in indexed.references:
                        self._references.setdefault(ref.name, []).append(ref)

    def symbols(self, name: str | None = None) -> list[Symbol]:
        """Return all indexed symbols, optionally filtered by name."""
        if name is None:
            return [s for syms in self._symbols.values() for s in syms]
        return list(self._symbols.get(name, []))

    def references(self, name: str | None = None) -> list[Reference]:
        """Return all indexed references, optionally filtered by name."""
        if name is None:
            return [r for refs in self._references.values() for r in refs]
        return list(self._references.get(name, []))

    def files_using(self, name: str) -> list[Path]:
        """Return files that define or reference ``name``."""
        files: set[Path] = set()
        for sym in self.symbols(name):
            files.add(Path(sym.file))
        for ref in self.references(name):
            files.add(Path(ref.file))
        return sorted(files)

    def related_symbols(self, name: str) -> list[str]:
        """Return other symbols that frequently co-occur in files with ``name``."""
        files = {str(p) for p in self.files_using(name)}
        counts: dict[str, int] = {}
        for sym in self.symbols():
            if sym.name == name:
                continue
            if sym.file in files:
                counts[sym.name] = counts.get(sym.name, 0) + 1
        return sorted(counts, key=lambda k: counts[k], reverse=True)[:20]

    def _should_skip(self, path: Path) -> bool:
        for part in path.parts:
            if part in self.SKIP_DIRS:
                return True
        try:
            if path.stat().st_size > 2 * 1024 * 1024:
                return True
        except Exception:
            return True
        return False

    def _index_file(self, path: Path) -> IndexedFile | None:
        try:
            source = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return None

        if path.suffix == ".py":
            return self._index_python(path, source)
        return self._index_generic(path, source)

    def _index_python(self, path: Path, source: str) -> IndexedFile:
        symbols: list[Symbol] = []
        references: list[Reference] = []

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return IndexedFile(path=path)

        class _StackedSymbolVisitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.stack: list[ast.AST] = []

            def _in_class(self) -> bool:
                return any(isinstance(n, ast.ClassDef) for n in self.stack)

            def visit(self, node: ast.AST) -> None:
                self.stack.append(node)
                super().visit(node)
                self.stack.pop()

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                kind = "method" if self._in_class() else "function"
                symbols.append(
                    Symbol(
                        name=node.name,
                        kind=kind,
                        file=str(path),
                        line=node.lineno,
                        col=node.col_offset,
                    )
                )
                self.generic_visit(node)

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                kind = "method" if self._in_class() else "function"
                symbols.append(
                    Symbol(
                        name=node.name,
                        kind=kind,
                        file=str(path),
                        line=node.lineno,
                        col=node.col_offset,
                    )
                )
                self.generic_visit(node)

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                symbols.append(
                    Symbol(
                        name=node.name,
                        kind="class",
                        file=str(path),
                        line=node.lineno,
                        col=node.col_offset,
                    )
                )
                self.generic_visit(node)

            def visit_Name(self, node: ast.Name) -> None:
                if isinstance(node.ctx, ast.Load):
                    references.append(
                        Reference(
                            name=node.id,
                            file=str(path),
                            line=node.lineno,
                            col=node.col_offset,
                        )
                    )

            def visit_Attribute(self, node: ast.Attribute) -> None:
                if isinstance(node.ctx, ast.Load):
                    references.append(
                        Reference(
                            name=node.attr,
                            file=str(path),
                            line=node.lineno,
                            col=node.col_offset,
                        )
                    )
                self.generic_visit(node)

        _StackedSymbolVisitor().visit(tree)
        return IndexedFile(path=path, symbols=symbols, references=references)

    def _index_generic(self, path: Path, source: str) -> IndexedFile:
        """Fallback regex-based indexing for non-Python languages."""
        symbols: list[Symbol] = []
        references: list[Reference] = []

        # Very rough patterns for class/function definitions.
        class_re = re.compile(
            r"^\s*(?:export\s+)?(?:class|struct|interface)\s+(\w+)", re.M
        )
        func_re = re.compile(
            r"^\s*(?:export\s+)?(?:function|def|fn|func)\s+(\w+)\s*\(", re.M
        )
        method_re = re.compile(r"^\s+(\w+)\s*\([^)]*\)\s*\{?\s*$", re.M)

        for match in class_re.finditer(source):
            line = source[: match.start()].count("\n") + 1
            symbols.append(
                Symbol(name=match.group(1), kind="class", file=str(path), line=line)
            )
        for match in _GENERIC_FUNC_RE.finditer(source):
            line = source[: match.start()].count("\n") + 1
            symbols.append(
                Symbol(name=match.group(1), kind="function", file=str(path), line=line)
            )
        for match in _GENERIC_METHOD_RE.finditer(source):
            line = source[: match.start()].count("\n") + 1
            symbols.append(
                Symbol(name=match.group(1), kind="method", file=str(path), line=line)
            )

        # Collect all word-like identifiers as references.
        for match in re.finditer(r"[A-Za-z_]\w*", source):
            line = source[: match.start()].count("\n") + 1
            references.append(
                Reference(
                    name=match.group(0),
                    file=str(path),
                    line=line,
                    col=match.start() - source.rfind("\n", 0, match.start()) - 1,
                )
            )

        return IndexedFile(path=path, symbols=symbols, references=references)
