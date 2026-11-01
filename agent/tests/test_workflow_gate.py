"""流程门 — spec → plan → 开工, 不靠自觉.

规则 (可执行):
- 每个 plan 头部必须声明 ``spec: `docs/staging/specs/<file>.md` ``;
  未归档 spec 的 legacy plan 必须显式写 ``workflow-exempt: <原因>``.
- 任务行 ``- [ ] T<n>:`` / ``- [x] T<n>:`` 的块内必须有
  ``spec: `#<section>```, 且该节在声明的 spec 里真实存在.
- 每个 spec 必须有边界节 (``非目标`` / ``deferred`` / ``failure modes``).
- 执行面: plan 头部声明 ``execution: inline|subagent|mixed``, 任务可用
  ``exec: inline|subagent`` 覆盖; ``mixed`` 下每个任务必须显式声明.
- 派发前提: ``[parallel]`` 任务的文件集必须两两不交; ``subagent`` 任务
  必须带 ``files`` 与 ``acceptance`` (子代理需独立完成并自验).

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
_EXEC_DECL_RE = re.compile(r"^execution:\s+(\S+)", re.MULTILINE)
_TASK_EXEC_RE = re.compile(r"^\s+exec:\s+(\S+)")
_TASK_FILES_RE = re.compile(r"^\s+files:\s+(.*)$")
_TASK_ACCEPT_RE = re.compile(r"^\s+acceptance:")
_PARALLEL_TASK_RE = re.compile(r"^- \[[ x]\] \[parallel\] ")
_BACKTICK_RE = re.compile(r"`([^`]+)`")
_EXEC_MODES = {"inline", "subagent", "mixed"}
_TASK_EXEC_MODES = {"inline", "subagent"}


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


def _field(block: list[str], pattern: re.Pattern[str]) -> re.Match[str] | None:
    for line in block:
        m = pattern.match(line)
        if m:
            return m
    return None


def validate_execution(plans_dir: Path = STAGING / "plans") -> list[str]:
    """执行面: 模式必须显式声明, 派发前提必须成立."""
    errors: list[str] = []
    for plan in sorted(plans_dir.glob("*.md")):
        text = plan.read_text(encoding="utf-8")
        if _EXEMPT_RE.search(text):
            continue
        decl = _EXEC_DECL_RE.search(text)
        if decl is None:
            errors.append(
                f"plan 未声明 execution (inline|subagent|mixed): {plan.name}"
            )
            continue
        default_mode = decl.group(1).strip("`")
        if default_mode not in _EXEC_MODES:
            errors.append(f"{plan.name}: execution 未知模式 {default_mode}")
            continue

        parallel_files: list[tuple[str, set[str]]] = []
        for task_id, block in _task_blocks(text):
            mode_m = _field(block, _TASK_EXEC_RE)
            task_mode = mode_m.group(1).strip("`") if mode_m else None
            if task_mode is not None and task_mode not in _TASK_EXEC_MODES:
                errors.append(f"{plan.name} {task_id}: exec 未知模式 {task_mode}")
                continue
            if default_mode == "mixed" and task_mode is None:
                errors.append(
                    f"{plan.name} {task_id}: execution=mixed 下必须显式声明 exec"
                )
            effective = task_mode or default_mode

            files_m = _field(block, _TASK_FILES_RE)
            files = set(_BACKTICK_RE.findall(files_m.group(1))) if files_m else set()

            if effective == "subagent":
                if not files:
                    errors.append(
                        f"{plan.name} {task_id}: exec=subagent 必须声明 files, 否则子代理无法独立完成"
                    )
                if _field(block, _TASK_ACCEPT_RE) is None:
                    errors.append(
                        f"{plan.name} {task_id}: exec=subagent 必须声明 acceptance, 否则无法自验"
                    )

            if _PARALLEL_TASK_RE.match(block[0]) and files:
                for other_id, other_files in parallel_files:
                    shared = files & other_files
                    if shared:
                        errors.append(
                            f"{plan.name} {task_id}/{other_id}: 并行任务共享文件 "
                            f"{sorted(shared)}, 不得并行派发"
                        )
                parallel_files.append((task_id, files))
    return errors


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
    errors.extend(validate_execution(plans_dir))
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
        "spec: `specs/s.md`\n"
        "execution: inline\n\n"
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


def test_execution_gate_rejects_unknown_mode_and_bad_dispatch(tmp_path):
    plans = tmp_path / "plans"
    plans.mkdir()
    (plans / "unknown.md").write_text("execution: parallel\n", encoding="utf-8")
    (plans / "mixed.md").write_text(
        "execution: mixed\n\n- [ ] T1: 未声明 exec\n  files: `a.py`\n",
        encoding="utf-8",
    )
    (plans / "parallel.md").write_text(
        "execution: inline\n\n"
        "- [ ] [parallel] T1: 一\n  files: `a.py`、`shared.py`\n"
        "- [ ] [parallel] T2: 二\n  files: `b.py`、`shared.py`\n",
        encoding="utf-8",
    )
    (plans / "subagent.md").write_text(
        "execution: subagent\n\n- [ ] T1: 缺 acceptance\n  files: `a.py`\n",
        encoding="utf-8",
    )
    errors = validate_execution(plans)
    assert any("execution 未知模式 parallel" in e for e in errors)
    assert any("mixed 下必须显式声明 exec" in e for e in errors)
    assert any("共享文件" in e for e in errors)
    assert any("必须声明 acceptance" in e for e in errors)
    # 合规的混合声明不报错.
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "ok.md").write_text(
        "execution: mixed\n\n"
        "- [ ] T1: 内联\n  exec: inline\n"
        "- [ ] T2: 子代理\n  exec: subagent\n  files: `b.py`\n  acceptance: 2 passed\n",
        encoding="utf-8",
    )
    assert validate_execution(clean) == []
