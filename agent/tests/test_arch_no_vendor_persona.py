# governance-gate
"""交付面不得把 agent 绑定到某个模型的实验人格名.

历史问题: 特定厂商的实验人格名一度写进报告标题、工具描述、错误文案、技能描述,
读起来像 agent 是为那一个模型/那一次实验而开发. 本门把"不再绑定"变成可执行断言:

- 扫描面 = 交付代码与文档 (agent/ 全部 + examples/ 去掉 out/ + 仓库 docs/ +
  顶层配置与审计文档);
- 历史运行证据 (research_outputs/、research/<run>/、examples/out/) **不在扫描面** ——
  它们是不可篡改的实验档案, 不能为了措辞回溯改日志;
- provider 标识 (如 ``internlm``) 与厂商名 (如 InternLM) 是**可选接入**, 不属人格名,
  允许保留.

被禁词用 Unicode 转义书写, 本文件自身不含字面量, 避免自指.
"""

from __future__ import annotations

import re
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = AGENT_ROOT.parent

#: 被禁的实验人格名 (Unicode 转义: 避免本门自己命中自己).
BANNED = "\u4e66\u751f"
_BANNED_RE = re.compile(BANNED)

_SCAN_ROOTS = (
    AGENT_ROOT / "huginn",
    AGENT_ROOT / "skills",
    AGENT_ROOT / "docs",
    AGENT_ROOT / "scripts",
    AGENT_ROOT / "tests",
    REPO_ROOT / "docs",
    REPO_ROOT / "examples",
    REPO_ROOT / "research",
)
_SCAN_FILES = (
    REPO_ROOT / "control_surface_audit.md",
    REPO_ROOT / "huginn.toml.example",
    REPO_ROOT / ".gitignore",
)
_SUFFIXES = {".py", ".md", ".toml", ".example", ".txt", ".yml", ".yaml", ".json"}


def _iter_files(root: Path):
    """root 下应扫描的文件; research/ 只扫顶层 (子目录是运行证据)."""
    if root.is_file():
        yield root
        return
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in {"__pycache__", "out", "shusheng_run"} for part in path.parts):
            continue
        if root.name == "research" and path.parent != root:
            continue
        if path.suffix.lower() not in _SUFFIXES and path.name != ".gitignore":
            continue
        yield path


def _scan(paths: list[Path]) -> list[str]:
    out: list[str] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if _BANNED_RE.search(line):
                try:
                    rel = path.relative_to(REPO_ROOT).as_posix()
                except ValueError:  # 自检用 tmp_path, 不在仓内
                    rel = path.as_posix()
                out.append(f"{rel}:{lineno}")
    return sorted(set(out))


def offenders(root_override: list[Path] | None = None) -> list[str]:
    """返回命中行; ``root_override`` 仅供自检使用."""
    if root_override is not None:
        return _scan(root_override)
    paths: list[Path] = []
    for root in _SCAN_ROOTS:
        paths.extend(_iter_files(root))
    paths.extend(_SCAN_FILES)
    return _scan(paths)


def test_no_vendor_persona_in_deliverable_surface():
    """交付面零命中: 报告/工具描述/错误文案/技能/文档不得出现实验人格名."""
    assert offenders() == []


def test_offender_scan_primitive(tmp_path: Path):
    """自检: 命中行能被检出, 且历史证据目录不会被误扫."""
    hit = tmp_path / "note.md"
    hit.write_text(BANNED + "\n", encoding="utf-8")
    clean = tmp_path / "clean.md"
    clean.write_text("模型\n", encoding="utf-8")
    errors = offenders(root_override=[hit, clean])
    assert len(errors) == 1
    assert errors[0].endswith("note.md:1")
