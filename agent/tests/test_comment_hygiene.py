"""注释卫生门 — 注释只解释「为什么/约束」, 不写变更史与任务号.

规则 (可执行, 不靠自觉):
- 禁止 changelog 式注释: "以前/旧实现/早期/demo 阶段/历史上" 等叙述历史;
- 禁止任务号考古: "P4-2:"、"P1-3:" 这类编号离开计划文档后不可追溯;
- 历史与取舍写进 plan/spec 或 PR 描述, 不写进代码;
- 例外: 注释含 ``comment-ok`` 显式标记 (且必须写明原因).

基线只减不增: 现存 151 处 / 62 文件是待清理存量, 新增一处即红;
清理后请下调 ``_BASELINE`` / ``_FILE_BASELINE``.
"""

from __future__ import annotations

import io
import re
import tokenize
from pathlib import Path

HUGINN_ROOT = Path(__file__).resolve().parents[1] / "huginn"

# 高精度标记: 只抓明确的"叙述历史"与"任务号", 避免误伤普通说明.
_ARCHAEOLOGY = re.compile(
    r"以前|旧实现|旧行为|旧版本|旧写法|demo 阶段|demo 写法|早期|曾经|原来"
    r"|回锁|已废弃|历史上|考古|MECE 审计记|spec [A-Z]\b|第0批"
    r"|(?<![A-Za-z0-9_])P\d+-\d+(?![A-Za-z0-9_])"
)

_ALLOW_MARKER = "comment-ok"

# 存量基线 (只减不增). 当前实测 151 处 / 62 文件.
_BASELINE = 151
_FILE_BASELINE = 62


def _file_hits(path: Path) -> list[tuple[int, str]]:
    src = path.read_text(encoding="utf-8")
    hits: list[tuple[int, str]] = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type != tokenize.COMMENT:
            continue
        # 只扫注释: 字符串/文档串里出现任务号不算违规 (可能是真实数据).
        if _ALLOW_MARKER in tok.string:
            continue
        if _ARCHAEOLOGY.search(tok.string):
            hits.append((tok.start[0], tok.string.strip()))
    return hits


def scan_archaeology(root: Path = HUGINN_ROOT) -> list[tuple[str, int, str]]:
    """返回 [(相对路径, 行号, 注释文本)], 按路径与行号排序."""
    out: list[tuple[str, int, str]] = []
    for path in sorted(root.rglob("*.py")):
        for lineno, text in _file_hits(path):
            out.append((str(path.relative_to(root)), lineno, text))
    return out


def test_no_new_history_or_task_id_comments():
    hits = scan_archaeology()
    files = {h[0] for h in hits}
    sample = "\n".join(f"  {p}:{n}: {t}" for p, n, t in hits[:5])
    assert len(hits) <= _BASELINE, (
        f"注释考古标记 {len(hits)} 处, 超过基线 {_BASELINE}:\n{sample}\n"
        "把历史写进 plan/spec, 注释只解释为什么与约束; 确需保留时加 comment-ok 并写明原因."
    )
    assert len(files) <= _FILE_BASELINE, (
        f"含考古注释的文件 {len(files)} 个, 超过基线 {_FILE_BASELINE}: "
        f"{sorted(files)[:5]}"
    )


def test_scanner_ignores_strings_and_honours_exemption(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text(
        'X = "P4-2: 这是数据, 不是注释"\n'
        "# P4-2: 这是注释, 应被抓住\n"
        "# comment-ok: 这是一段必须保留的历史引文\n",
        encoding="utf-8",
    )
    hits = scan_archaeology(tmp_path)
    assert len(hits) == 1
    assert hits[0][1] == 2
