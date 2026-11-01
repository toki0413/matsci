# governance-gate
"""治理门自标记与归属台账 (防"改名即逃逸").

治理门此前只靠两件事被识别: 文件名约定 (``tests/test_arch_*.py``) 与手工登记表
(``governance_audit.FULL_SUITE_ONLY`` / ``GOVERNANCE_META_TESTS``). 两者都有同一个
漏洞 —— 把门改个名字 (``test_arch_x.py`` → ``test_gate_x.py``), 孤儿门检测/登记面
全部失效, 门从此游离在治理体系之外仍全绿. ``test_doc_assertions.py`` 就是活例:
它是治理门, 但不在任何 guard step、也不在登记表里, 无任何机制能发现.

本门把"这是治理门"的事实写进门的**自身** (首行 ``# governance-gate``), 再做双向闭合:

1. 带标记的文件 == 登记集合 (arch 族 ∪ FULL_SUITE_ONLY ∪ GOVERNANCE_META_TESTS);
   标记了未登记 → 红; 登记了没标记 → 红 (防删标记逃逸).
2. 引用治理锚点 (governance_audit / doc_assertions / SURFACE_SPECS /
   KIND_HANDLE_SPECS / BUNDLE_MANIFEST / silent_exceptions_baseline) 的测试文件
   必须带标记 —— 这是**默认发现**, 不靠作者记得去登记.
3. 每个带标记的门必须有归属: 在 FULL_SUITE_ONLY, 或被任一 workflow 的 run 文本
   精确引用 (guard step).

weightless: 按文件路径加载 governance_audit, 不走 huginn.cli 包 __init__.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = AGENT_ROOT.parent
MARKER = "# governance-gate"
#: 治理锚点: 引用这些符号的测试一定是治理门, 必须显式标记.
_GOVERNANCE_ANCHORS = re.compile(
    r"governance_audit|doc_assertions|SURFACE_SPECS|KIND_HANDLE_SPECS"
    r"|BUNDLE_MANIFEST|silent_exceptions_baseline"
)

_GOV = AGENT_ROOT / "huginn" / "cli" / "governance_audit.py"
_ga_spec = importlib.util.spec_from_file_location("governance_audit", _GOV)
_ga = importlib.util.module_from_spec(_ga_spec)
_ga_spec.loader.exec_module(_ga)


def _all_test_files(root: Path = AGENT_ROOT) -> list[Path]:
    """``tests/**/test_*.py``, 跳过 ``__pycache__``."""
    return sorted(
        p
        for p in (root / "tests").rglob("test_*.py")
        if "__pycache__" not in p.parts
    )


def _rel(path: Path, root: Path = AGENT_ROOT) -> str:
    return path.relative_to(root).as_posix()


def _is_marked(path: Path) -> bool:
    """首 3 行内出现标记 (容许 shebang / 编码行)."""
    lines = path.read_text(encoding="utf-8").splitlines()[:3]
    return any(line.strip() == MARKER for line in lines)


def registered_gates(root: Path = AGENT_ROOT) -> set[str]:
    """登记集合 = arch 族 ∪ FULL_SUITE_ONLY ∪ GOVERNANCE_META_TESTS."""
    out = set(_ga.discover_arch_tests(root))
    out |= {t for t in _ga.FULL_SUITE_ONLY if (root / t).exists()}
    out |= {t for t in _ga.GOVERNANCE_META_TESTS if (root / t).exists()}
    return out


def marked_gates(root: Path = AGENT_ROOT) -> set[str]:
    """带 ``# governance-gate`` 标记的测试文件."""
    return {_rel(p, root) for p in _all_test_files(root) if _is_marked(p)}


def governance_referencing_files(root: Path = AGENT_ROOT) -> set[str]:
    """引用了治理锚点、却缺标记的文件 (应全部带标记)."""
    out = set()
    for p in _all_test_files(root):
        if _is_marked(p):
            continue
        if _GOVERNANCE_ANCHORS.search(p.read_text(encoding="utf-8")):
            out.add(_rel(p, root))
    return out


def _workflow_texts(repo_root: Path = REPO_ROOT) -> list[str]:
    wf = repo_root / ".github" / "workflows"
    if not wf.is_dir():
        return []
    return [p.read_text(encoding="utf-8", errors="replace") for p in wf.glob("*.yml")]


def owners_of(rel: str, repo_root: Path = REPO_ROOT) -> list[str]:
    """该门被哪些 workflow 的 run 文本引用 (精确路径匹配)."""
    return [
        f".github/workflows/{p.name}"
        for p in (repo_root / ".github" / "workflows").glob("*.yml")
        if rel in p.read_text(encoding="utf-8", errors="replace")
    ]


def gates_without_owner(root: Path = AGENT_ROOT, repo_root: Path = REPO_ROOT) -> list[str]:
    """带标记却既不在 FULL_SUITE_ONLY、也未被任何 workflow 引用的门."""
    out = []
    for rel in sorted(marked_gates(root)):
        if rel in set(_ga.FULL_SUITE_ONLY):
            continue
        if owners_of(rel, repo_root):
            continue
        out.append(rel)
    return out


def test_every_marked_gate_is_registered():
    """标记了却未登记: 治理门游离在台账之外."""
    assert sorted(marked_gates() - registered_gates()) == []


def test_every_registered_gate_is_marked():
    """登记了却没标记: 删掉标记即可绕过本门 (双向闭合的另一半)."""
    assert sorted(registered_gates() - marked_gates()) == []


def test_governance_referencing_tests_are_marked():
    """引用治理锚点的测试必须自带标记 (默认发现, 非纯 opt-in)."""
    assert sorted(governance_referencing_files()) == []


def test_every_marked_gate_has_owner():
    """每个治理门都要有归属: full-suite-only 或某 workflow guard."""
    assert gates_without_owner() == []


def test_marker_detection_primitive(tmp_path: Path):
    """自检: 标记只在首 3 行生效, 正文深处出现同字样不算 (防门自己失效)."""
    marked = tmp_path / "test_marked.py"
    marked.write_text(f"{MARKER}\n\"\"\"doc\"\"\"\n", encoding="utf-8")
    deep = tmp_path / "test_deep.py"
    deep.write_text('"""doc"""\n# ' + MARKER + "\n", encoding="utf-8")
    assert _is_marked(marked) is True
    assert _is_marked(deep) is False
