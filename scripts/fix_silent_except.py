"""AST-driven, line-based fixer for truly-silent except handlers.

For each truly-silent handler (body is just pass/return<const>/continue/break
or a single `var = None` assignment, with no logging/print/reraise):

- If body is [Pass]: replace the `pass` line with
      logger.debug("best-effort op failed", exc_info=True)
- Else: insert a new line BEFORE the first body statement with the body's
  indentation:
      logger.debug("best-effort op failed", exc_info=True)

Only touches files that already have a module-level `logger` binding.
Skips: tests/, plugins/science-skills/ (vendored), __pycache__/, .venv/.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

LOG_METHODS = {"debug", "info", "warning", "warn", "error", "exception", "critical", "log"}
SIMPLE_CONSTS = (ast.Constant,)


def _is_logging_call(node: ast.AST) -> bool:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        return node.func.attr in LOG_METHODS
    return False


def _is_print_call(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print"


def _has_reraise(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Raise) and child.exc is None:
            return True
    return False


def _body_is_truly_silent(body: list[ast.stmt]) -> bool:
    if not body or len(body) > 2:
        return False
    for stmt in body:
        if isinstance(stmt, ast.Pass):
            continue
        if isinstance(stmt, ast.Return):
            if stmt.value is None:
                continue
            v = stmt.value
            if isinstance(v, SIMPLE_CONSTS):
                continue
            if isinstance(v, ast.Name) and v.id in {"None", "True", "False"}:
                continue
            if isinstance(v, (ast.List, ast.Tuple, ast.Dict, ast.Set)):
                elts = getattr(v, "elts", None) or getattr(v, "keys", None)
                if elts is not None and len(elts) == 0:
                    continue
            return False
        if isinstance(stmt, (ast.Continue, ast.Break)):
            continue
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                return False
            if isinstance(stmt.value, SIMPLE_CONSTS):
                continue
            if isinstance(stmt.value, ast.Name) and stmt.value.id in {"None", "True", "False"}:
                continue
            return False
        return False
    # Final guard: no logging/print anywhere in body
    for stmt in body:
        for child in ast.walk(stmt):
            if _is_logging_call(child) or _is_print_call(child):
                return False
    return True


def _file_has_module_logger(tree: ast.Module) -> bool:
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "logger":
                    return True
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "logger":
            return True
    return False


def fix_file(path: Path, dry_run: bool = False) -> int:
    """Returns number of handlers fixed. Edits file in place unless dry_run."""
    try:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return 0
    if not _file_has_module_logger(tree):
        return 0

    # Collect handlers to fix, sorted by line descending so insertions don't
    # shift later line numbers.
    fixes: list[tuple[int, str, bool]] = []  # (line, indent, replace_pass)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        if _has_reraise(node):
            continue
        if not _body_is_truly_silent(node.body):
            continue
        if any(_is_logging_call(c) or _is_print_call(c) for c in ast.walk(node)):
            continue
        first = node.body[0]
        # Indentation = leading whitespace of the first body statement's line
        line_idx = first.lineno - 1
        line = src.splitlines()[line_idx] if line_idx < len(src.splitlines()) else ""
        indent = line[: len(line) - len(line.lstrip())]
        replace_pass = isinstance(first, ast.Pass) and len(node.body) == 1
        fixes.append((first.lineno, indent, replace_pass))

    if not fixes:
        return 0

    lines = src.splitlines(keepends=True)
    # Apply fixes from bottom to top so line numbers stay valid
    fixes.sort(key=lambda x: -x[0])
    new_stmt_body = 'logger.debug("best-effort op failed", exc_info=True)'
    for lineno, indent, replace_pass in fixes:
        idx = lineno - 1
        if replace_pass:
            # Replace the `pass` line, but preserve any trailing inline comment.
            original = lines[idx]
            ending = "\n" if original.endswith("\n") else ""
            # Extract trailing comment from the `pass` line, if any.
            content = original.rstrip("\r\n")
            # Strip indent + `pass` to find what follows
            stripped = content.lstrip()
            # stripped starts with "pass"; capture remainder after "pass"
            after_pass = stripped[4:].strip()  # may be "" or "# comment"
            comment = after_pass if after_pass.startswith("#") else ""
            if comment:
                lines[idx] = f"{indent}{new_stmt_body}  {comment}{ending}"
            else:
                lines[idx] = f"{indent}{new_stmt_body}{ending}"
        else:
            # Insert before this line
            lines.insert(idx, f"{indent}{new_stmt_body}\n")

    new_src = "".join(lines)
    # Sanity: must still parse
    try:
        ast.parse(new_src, filename=str(path))
    except SyntaxError as e:
        print(f"  SKIP (would break syntax): {path}: {e}", file=sys.stderr)
        return 0

    if not dry_run:
        path.write_text(new_src, encoding="utf-8")
    return len(fixes)


def main(roots: list[str], dry_run: bool = False) -> None:
    total_files = 0
    total_fixes = 0
    skipped = 0
    for root in roots:
        for path in Path(root).rglob("*.py"):
            s = str(path)
            if "/.venv/" in s or "/__pycache__/" in s or "/plugins/science-skills/" in s:
                continue
            n = fix_file(path, dry_run=dry_run)
            if n > 0:
                total_files += 1
                total_fixes += n
                print(f"  {n:3d}  {path}")
    mode = "DRY RUN" if dry_run else "APPLIED"
    print(f"\n[{mode}] {total_fixes} handlers across {total_files} files")


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    roots = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(roots or ["huginn"], dry_run=dry)
