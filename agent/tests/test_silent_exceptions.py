# governance-gate
"""静默异常门 — 真正被吞掉的失败必须显式降级并留痕.

口径 (AST, 覆盖 ``huginn/`` 全树):
- handler 体 ≤2 条语句;
- 体内没有 raise / warning 及以上日志 / print, 且只有 pass、return 常量、
  continue/break、简单哨兵赋值 —— 即失败在常规日志里完全不可见.

基线只减不增: 现存 427 处 / 224 文件是待清理存量, 新增一处即红.
要保留一处静默处理时, 至少在 handler 里记一条 debug/warning 日志说明降级,
或写成 ``contextlib.suppress`` (not an ExceptHandler, 本门不计).
"""

from __future__ import annotations

import ast
from pathlib import Path

HUGINN_ROOT = Path(__file__).resolve().parents[1] / "huginn"

_LOUD_LOG_METHODS = frozenset({"warning", "warn", "error", "exception", "critical", "fatal"})
_SIMPLE_CONST_STMTS = (ast.Pass, ast.Continue, ast.Break)

# 存量基线 (按文件, 只减不增). 单一事实来源是同目录的 JSON 快照:
# 全局总数此前是一个标量 —— A 模块从 5 涨到 50、B 模块从 50 降到 5, 总数不变就绿,
# 棘轮被"跨模块腾挪"绕开. 现在每个文件各自持有上限, 单点冒头即红.
_BASELINE_PATH = Path(__file__).with_name("silent_exceptions_baseline.json")


def _baseline() -> dict[str, int]:
    import json

    return json.loads(_BASELINE_PATH.read_text(encoding="utf-8"))


_BY_FILE = _baseline()
_BASELINE = sum(_BY_FILE.values())
_FILE_BASELINE = len(_BY_FILE)


def _has_visibility(stmt: ast.stmt) -> bool:
    """语句里是否有 raise / warning 级日志 / print."""
    for child in ast.walk(stmt):
        if isinstance(child, ast.Raise):
            return True
        if isinstance(child, ast.Call):
            if isinstance(child.func, ast.Attribute) and child.func.attr in _LOUD_LOG_METHODS:
                return True
            if isinstance(child.func, ast.Name) and child.func.id == "print":
                return True
    return False


def _is_simple(stmt: ast.stmt) -> bool:
    if isinstance(stmt, _SIMPLE_CONST_STMTS):
        return True
    if isinstance(stmt, ast.Return):
        v = stmt.value
        if v is None or isinstance(v, ast.Constant):
            return True
        return isinstance(v, ast.List | ast.Tuple | ast.Dict | ast.Set) and not getattr(
            v, "elts", getattr(v, "keys", [])
        )
    if isinstance(stmt, ast.Assign):
        if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
            return False
        v = stmt.value
        return isinstance(v, ast.Constant) or (
            isinstance(v, ast.Name) and v.id in {"None", "True", "False"}
        )
    return False


def _is_truly_silent(node: ast.ExceptHandler) -> bool:
    if not node.body or len(node.body) > 2:
        return False
    return all(_is_simple(s) and not _has_visibility(s) for s in node.body)


def scan_truly_silent(root: Path = HUGINN_ROOT) -> list[tuple[str, int]]:
    """返回 [(相对路径, 行号)]."""
    out: list[tuple[str, int]] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and _is_truly_silent(node):
                out.append((str(path.relative_to(root)), node.lineno))
    return out


def test_no_new_truly_silent_handlers():
    hits = scan_truly_silent()
    files = {h[0] for h in hits}
    sample = "\n".join(f"  {p}:{n}" for p, n in hits[:8])
    assert len(hits) <= _BASELINE, (
        f"真正静默的 except 处理器 {len(hits)} 个, 超过基线 {_BASELINE}:\n{sample}\n"
        "能跳过就写 contextlib.suppress; 要降级就记 warning 说明原因; 不能吞就别捕获."
    )
    assert len(files) <= _FILE_BASELINE, (
        f"含静默 except 的文件 {len(files)} 个, 超过基线 {_FILE_BASELINE}: "
        f"{sorted(files)[:5]}"
    )


def test_per_file_silent_handlers_only_shrink():
    """单文件棘轮: 某文件静默异常数不得超过该文件基线 (防跨模块腾挪)."""
    from collections import Counter

    live = Counter(path for path, _ in scan_truly_silent())
    over = {
        path: (count, _BY_FILE.get(path, 0))
        for path, count in live.items()
        if count > _BY_FILE.get(path, 0)
    }
    assert not over, (
        f"以下文件的静默 except 超出各自基线 (实测, 上限): {over} —— "
        "跨模块腾挪不算收敛; 只降不增, 确需新增须在 "
        f"{_BASELINE_PATH.name} 显式抬高该文件上限 (可见决策)."
    )


def test_scanner_ignores_logged_and_reraising_handlers(tmp_path):
    src = tmp_path / "probe.py"
    src.write_text(
        "import logging\n"
        "log = logging.getLogger(__name__)\n"
        "try:\n"
        "    x = 1\n"
        "except Exception:\n"
        "    pass\n"
        "try:\n"
        "    y = 1\n"
        "except Exception:\n"
        "    log.debug('best effort')\n"
        "try:\n"
        "    z = 1\n"
        "except Exception:\n"
        "    raise\n",
        encoding="utf-8",
    )
    hits = scan_truly_silent(tmp_path)
    assert hits == [("probe.py", 5)]
