"""skill_importer 的最小自检：三套格式互相映射 + 导出回 Huginn 格式。

直接 `python test_skill_importer.py` 也能跑（不依赖 pytest），用 assert 兜底。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from huginn.plugins.skill_importer import SkillImporter


OPENCLAW_SKILL = """\
---
name: oc_demo
description: openclaw demo
tools:
  - shell
  - editor
trigger_conditions:
  - "run tests"
  - "check coverage"
steps:
  - Analyze project
  - Run pytest
---
# OpenClaw demo body
"""

HERMES_SKILL = """\
---
name: hermes_demo
description: hermes demo
trigger: "写单元测试"
steps:
  - Identify patterns
  - Generate cases
tags: [python, testing]
---
# Hermes demo body
"""

HUGINN_SKILL = """\
---
name: native_demo
description: native demo
allowed-tools: [shell]
when_to_use: "when relaxing"
paths: ["**/*.py"]
---
# Native body
"""


def _write(root: Path, text: str, name: str = "SKILL.md") -> Path:
    p = root / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def test_openclaw_mapping():
    with tempfile.TemporaryDirectory() as d:
        p = _write(Path(d), OPENCLAW_SKILL)
        s = SkillImporter().import_from_openclaw(p)
        assert s.name == "oc_demo"
        assert s.required_tools == ["shell", "editor"]
        assert s.metadata["when_to_use"] == "run tests OR check coverage"
        assert len(s.steps) == 2
        assert s.metadata["platform"] == "openclaw"


def test_hermes_mapping():
    with tempfile.TemporaryDirectory() as d:
        p = _write(Path(d), HERMES_SKILL)
        s = SkillImporter().import_from_hermes(p)
        assert s.name == "hermes_demo"
        assert s.metadata["when_to_use"] == "写单元测试"
        assert s.tags == ["python", "testing"]
        assert s.metadata["tags"] == ["python", "testing"]
        assert len(s.steps) == 2


def test_auto_detect_and_directory():
    imp = SkillImporter()
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _write(root / "oc", OPENCLAW_SKILL)
        _write(root / "hm", HERMES_SKILL)
        _write(root / "hn", HUGINN_SKILL)
        skills = imp.import_directory(root, "auto")
        names = {s.name for s in skills}
        assert names == {"oc_demo", "hermes_demo", "native_demo"}
        plats = {s.name: s.metadata["platform"] for s in skills}
        assert plats["oc_demo"] == "openclaw"
        assert plats["hermes_demo"] == "hermes"
        assert plats["native_demo"] == "huginn"


def test_export_roundtrip():
    imp = SkillImporter()
    with tempfile.TemporaryDirectory() as d:
        src = _write(Path(d), OPENCLAW_SKILL)
        skill = imp.import_from_openclaw(src)
        out = Path(d) / "out" / "oc_demo.md"
        imp.export_to_huginn(skill, out)
        # 导出后再当 native 导回来，关键字段要对得上
        back = imp.import_file(out, "auto")
        assert back.name == "oc_demo"
        assert back.required_tools == ["shell", "editor"]
        assert back.metadata["when_to_use"] == "run tests OR check coverage"


if __name__ == "__main__":
    for fn in (
        test_openclaw_mapping,
        test_hermes_mapping,
        test_auto_detect_and_directory,
        test_export_roundtrip,
    ):
        fn()
        print(f"ok: {fn.__name__}")
    print("all skill_importer checks passed")
