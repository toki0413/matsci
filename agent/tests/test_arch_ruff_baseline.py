# governance-gate
"""全仓 lint 棘轮 (per-file only-shrink).

`ruff check .` 的存量违规按文件计数冻结在 `tests/ruff_baseline.json`:
  - 单文件违规数只能降, 不能升;
  - 基线里没有的文件出现任何违规 → 红 (新代码必须干净);
  - 总数只能降.
ruff 版本与基线不一致时 skip (规则集随版本变化, 计数不可比).
"""

from __future__ import annotations

import collections
import json
import subprocess
import sys
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = Path(__file__).with_name("ruff_baseline.json")


def _baseline() -> dict:
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def _normalize(filename: str, root: Path = AGENT_ROOT) -> str:
    """ruff 的 filename 可能是绝对路径; 统一成相对 agent/ 的 posix 路径."""
    p = Path(filename)
    if p.is_absolute():
        p = p.resolve().relative_to(root.resolve())
    return p.as_posix()


def _run_ruff(root: Path = AGENT_ROOT) -> tuple[str, dict[str, int]]:
    ver = subprocess.run(
        [sys.executable, "-m", "ruff", "--version"],
        capture_output=True, text=True, cwd=root,
    )
    if ver.returncode != 0:
        raise RuntimeError(ver.stderr.strip() or "ruff 未安装")
    proc = subprocess.run(
        [sys.executable, "-m", "ruff", "check", ".", "--output-format=json"],
        capture_output=True, text=True, cwd=root, encoding="utf-8",
    )
    rows = json.loads(proc.stdout or "[]")
    counts = collections.Counter(_normalize(r["filename"], root) for r in rows)
    return ver.stdout.strip().removeprefix("ruff "), dict(counts)


def _exceeds(actual: dict[str, int], baseline: dict[str, int]) -> dict[str, int]:
    """比基线更差的条目 (新文件或计数上涨)."""
    return {
        f: n for f, n in actual.items()
        if n > baseline.get(f, 0)
    }


def test_ruff_baseline_only_shrinks():
    """不变量: 逐文件违规数不涨, 且总数不涨."""
    base = _baseline()
    version, actual = _run_ruff()
    if version != base["ruff_version"]:
        import pytest

        pytest.skip(
            f"ruff 版本 {version} 与基线 {base['ruff_version']} 不同, "
            "规则集差异不可比; 升级 ruff 时请同步重生成 tests/ruff_baseline.json"
        )
    worse = _exceeds(actual, base["files"])
    assert worse == {}, (
        f"以下文件 lint 违规数高于基线: {worse} —— 修掉新增违规; "
        "存量收敛后用 `python -m ruff check . --output-format=json` 重生成基线"
    )
    assert sum(actual.values()) <= base["total"], (
        f"总违规数从 {base['total']} 涨到 {sum(actual.values())}"
    )


def test_exceeds_primitive():
    """自检: 新文件与计数上涨都能被检出 (防棘轮自己失效)."""
    base = {"a.py": 2}
    assert _exceeds({"a.py": 1}, base) == {}
    assert _exceeds({"a.py": 3}, base) == {"a.py": 3}
    assert _exceeds({"b.py": 1}, base) == {"b.py": 1}
