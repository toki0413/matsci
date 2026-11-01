"""流程门 — spec → plan → 开工, 不靠自觉.

规则 (可执行):
- 每个 plan 头部必须声明 ``spec: `docs/staging/specs/<file>.md` ``;
  未归档 spec 的 legacy plan 必须显式写 ``workflow-exempt: <原因>``.
- 任务行 ``- [ ] T<n>:`` / ``- [x] T<n>:`` 的块内必须有
  ``spec: `#<section>```, 且该节在声明的 spec 里真实存在.
- 每个 spec 必须有边界节 (``非目标`` / ``deferred`` / ``failure modes``).

这样"先写 spec, 再写 plan, 最后开工"的顺序留下可机器校验的锚点:
任务离开 spec 就没有开工许可.
"""

from __future__ import annotations

import re
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
STAGING = AGENT_ROOT / "docs" / "staging"

_BOUNDARY_MARKERS = (
    "非目标",
    "deferred",
    "failure modes",
    "失败模式",
    "non-goal",
    "out of scope",
)
_TASK_RE = re.compile(r"^- \[[ x]\] (?:\[parallel\] )?(T\d+):")
_SPEC_DECL_RE = re.compile(r"^spec:\s+`([^`]+)`", re.MULTILINE)
_EXEMPT_RE = re.compile(r"^workflow-exempt:\s+\S", re.MULTILINE)
_TASK_SPEC_RE = re.compile(r"^\s+spec:\s+`#([^`]+)`")


def _slug(heading: str) -> str:
    s = re.sub(r"[^\w\u4e00-\u9fff -]", "", heading.strip().lower())
    return re.sub(r"\s+", "-", s).strip("-")


def _spec_slugs(text: str) -> set[str]:
    return {_slug(line[3:]) for line in text.splitlines() if line.startswith("## ")}


def _task_blocks(text: str) -> list[tuple[str, list[str]]]:
    """按任务行切块; 块在下一个任务行或下一个标题处结束."""
    lines = text.splitlines()
    blocks: list[tuple[str, list[str]]] = []
    current: tuple[str, list[str]] | None = None
    for line in lines:
        m = _TASK_RE.match(line)
        if m:
            current = (m.group(1), [line])
            blocks.append(current)
            continue
        if line.startswith("## ") or line.startswith("# "):
            current = None
            continue
        if current is not None:
            current[1].append(line)
    return blocks


def validate(plans_dir: Path = STAGING / "plans", specs_dir: Path = STAGING / "specs") -> list[str]:
    """返回全部违规; 空列表 = 流程合规."""
    errors: list[str] = []

    specs: dict[Path, str] = {}
    for path in sorted(specs_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        specs[path] = text
        if not any(k in text.lower() for k in _BOUNDARY_MARKERS):
            errors.append(f"spec 缺边界节(非目标/deferred/failure modes): {path.name}")

    for plan in sorted(plans_dir.glob("*.md")):
        text = plan.read_text(encoding="utf-8")
        decl = _SPEC_DECL_RE.search(text)
        if not decl:
            if _EXEMPT_RE.search(text):
                continue
            errors.append(
                f"plan 未声明 spec 也未标 workflow-exempt: {plan.name}"
            )
            continue
        spec_path = (AGENT_ROOT / decl.group(1)).resolve()
        if not spec_path.exists():
            spec_path = (plans_dir.parent / decl.group(1)).resolve()
        if spec_path not in specs:
            errors.append(f"plan 声明的 spec 不存在或不在 staging/specs: {decl.group(1)}")
            continue
        slugs = _spec_slugs(specs[spec_path])
        for task_id, block in _task_blocks(text):
            anchor = next(
                (
                    m.group(1)
                    for line in block
                    if (m := _TASK_SPEC_RE.match(line))
                ),
                None,
            )
            if anchor is None:
                errors.append(f"{plan.name} {task_id}: 缺 spec 锚点, 不得开工")
            elif anchor not in slugs:
                errors.append(
                    f"{plan.name} {task_id}: 锚点 #{anchor} 不在 {spec_path.name}"
                )
    return errors


def test_all_plans_are_anchored_to_real_spec_sections():
    errors = validate()
    assert not errors, "\n".join(errors)


def test_gate_rejects_missing_anchor_ghost_anchor_and_no_boundary(tmp_path):
    specs = tmp_path / "specs"
    plans = tmp_path / "plans"
    specs.mkdir()
    plans.mkdir()
    (specs / "s.md").write_text("# S\n\n## A. 定义\n\n无边界\n", encoding="utf-8")
    (plans / "p.md").write_text(
        "spec: `specs/s.md`\n\n"
        "- [ ] T1: 无锚点\n"
        "  goal: x\n\n"
        "- [ ] T2: 幽灵锚点\n"
        "  spec: `#b-不存在`\n",
        encoding="utf-8",
    )
    errors = validate(plans, specs)
    assert any("缺 spec 锚点" in e for e in errors)
    assert any("不在 s.md" in e for e in errors)
    assert any("缺边界节" in e for e in errors)
