"""世界表征盘点文档防漂移自检.

<weltrepresentation-inventory.md> 是对照源码的审计快照. 让它"永远能对上代码"——
只要文档里引用的 huginn 路径或断言"真缺"的类名在代码里出现/消失, 本测试就红,
杜绝盘点再次退化成纸面。
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_DOC = _REPO / "docs/research-notes/world-representation-inventory.md"


def _doc_text() -> str:
    return _DOC.read_text(encoding="utf-8")


def test_doc_exists_and_registered_in_index():
    assert _DOC.exists()
    index = (_REPO / "docs/INDEX.md").read_text(encoding="utf-8")
    assert "world-representation-inventory.md" in index


def test_every_huginn_path_in_doc_exists():
    """文档引用的每个 huginn/**/*.py 路径必须真实存在"""
    src = _doc_text()
    paths = set(re.findall(r"huginn/[A-Za-z0-9_/]+\.py", src))
    assert paths, "文档未提取到任何 huginn 路径"
    missing = [p for p in sorted(paths) if not (_REPO / p).exists()]
    assert not missing, f"文档引用了不存在的源码路径: {missing}"


def test_true_gaps_are_still_gaps():
    """文档标'真缺'的符号(**表格结论列**)不得已在代码里出现(否则要改文档标为已实现)"""
    src_lines = _doc_text().splitlines()
    for sym in ("StateEstimator", "StateSnapshot", "ObsVector", "ForwardPredictor"):
        marked_gap = any(
            ln.startswith("|") and ("`" + sym + "`") in ln and "**真缺**" in ln
            for ln in src_lines
        )
        if not marked_gap:
            continue
        hits = _grep_symbol(sym)
        assert not hits, f"文档声称 {sym} 真缺, 但代码已出现: {hits}"


def _grep_symbol(symbol: str) -> list[str]:
    hits: list[str] = []
    for py in (_REPO / "huginn").rglob("*.py"):
        if py.name.startswith("__"):
            continue
        try:
            text = py.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if "class " + symbol in line:
                hits.append(f"{py.relative_to(_REPO)}:{lineno}")
    return hits


def test_stage0_components_exist():
    """盘点文档标'阶段0 已实现'的核心件必须真实可import(防回退为纸面)"""
    mod = (_REPO / "huginn/security/world_state.py").read_text(encoding="utf-8")
    for cls in ("ObsVector", "StateSnapshot", "StateEstimator", "ForwardPredictor"):
        assert "class " + cls in mod, f"world_state.py 缺少 class {cls}"
