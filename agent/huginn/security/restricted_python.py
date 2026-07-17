"""Restricted Python execution for the sandbox endpoint.

Pre-scans code with AST to block dangerous imports and builtins before
allowing execution in a subprocess sandbox.
"""

from __future__ import annotations

import ast
import re


class RestrictedPythonError(Exception):
    """Raised when code violates the restricted execution policy."""


# Modules that may compromise the host if imported
FORBIDDEN_MODULES: set[str] = {
    "os",
    "sys",
    "subprocess",
    "socket",
    "urllib",
    "urllib.request",
    "http",
    "http.client",
    "ftplib",
    "telnetlib",
    "smtplib",
    "poplib",
    "imaplib",
    "nntplib",
    "ctypes",
    "multiprocessing",
    "threading",
    "_thread",
    "builtins",
    "importlib",
    "pkgutil",
    "site",
    "pathlib",
    "shutil",
    "tempfile",
    "pickle",
    "marshal",
    "shelve",
}

# Built-in functions that must not be called
FORBIDDEN_BUILTINS: set[str] = {
    "__import__",
    "eval",
    "exec",
    "compile",
    "open",
    "input",
    "exit",
    "quit",
    "help",
}

# Dangerous dunder attributes
FORBIDDEN_ATTRIBUTES: set[str] = {
    "__subclasses__",
    "__bases__",
    "__base__",
    "__mro__",
    "__globals__",
    "__code__",
    "__closure__",
    "__defaults__",
    "__class__",
    "__dict__",
    "__import__",
    "__builtins__",
}


class _PolicyChecker(ast.NodeVisitor):
    """AST visitor that enforces the restricted execution policy."""

    def __init__(self) -> None:
        self.violations: list[str] = []

    def _report(self, node: ast.AST, message: str) -> None:
        self.violations.append(f"Line {node.lineno}: {message}")

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root in FORBIDDEN_MODULES:
                self._report(node, f"Forbidden import: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        if node.module:
            root = node.module.split(".")[0]
            if root in FORBIDDEN_MODULES:
                self._report(node, f"Forbidden import: {node.module}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        # Detect eval(), exec(), compile(), __import__(), open()
        if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_BUILTINS:
            self._report(node, f"Forbidden builtin call: {node.func.id}()")
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in FORBIDDEN_BUILTINS
        ):
            self._report(node, f"Forbidden method call: {node.func.attr}()")
        # G23: 拦截 getattr(obj, '__class__') 字符串参数逃逸
        # visit_Attribute 已拦 obj.__class__ 直接访问, 但 getattr 用字符串绕过
        # generic_visit 会递归访问内层 Call, 所以 getattr(getattr(...)) 也被覆盖
        if isinstance(node.func, ast.Name) and node.func.id == "getattr":
            if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                attr_name = node.args[1].value
                if isinstance(attr_name, str) and attr_name in FORBIDDEN_ATTRIBUTES:
                    self._report(
                        node,
                        f"Forbidden getattr access: .{attr_name}",
                    )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        if node.attr in FORBIDDEN_ATTRIBUTES:
            self._report(node, f"Forbidden attribute access: .{node.attr}")
        self.generic_visit(node)


def validate_code(code: str) -> None:
    """Validate Python code against the restricted execution policy.

    Raises RestrictedPythonError if the code contains forbidden constructs.
    """
    if not code or not code.strip():
        raise RestrictedPythonError("Empty code")

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise RestrictedPythonError(f"Syntax error: {e}") from e

    checker = _PolicyChecker()
    checker.visit(tree)

    if checker.violations:
        raise RestrictedPythonError(
            "Code violates restricted execution policy:\n"
            + "\n".join(f"  - {v}" for v in checker.violations)
        )
