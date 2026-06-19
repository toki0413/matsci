"""Tests for Nuwa-style persona skill loading."""

from __future__ import annotations

from pathlib import Path

from huginn.persona_loader import load_persona_skill, scan_persona_skills
from huginn.personas import PersonaManager

SAMPLE_SKILL = """---
name: paul-graham
description: Startup and writing perspective
when_to_use:
  - evaluating startup ideas
  - writing clearly
---

# Paul Graham

## Who Paul Graham Is
A founder, essayist, and investor.

## How Paul Graham Thinks
- Start with the problem, not the idea.
- Write simply and concretely.

## What Paul Graham Would NOT Do
- Chase trends without a real problem.
"""


def test_load_persona_skill_parses_frontmatter_and_body(tmp_path: Path) -> None:
    path = tmp_path / "paul-graham.md"
    path.write_text(SAMPLE_SKILL, encoding="utf-8")

    persona = load_persona_skill(path)
    assert persona.name == "paul-graham"
    assert persona.description == "Startup and writing perspective"
    assert "evaluating startup ideas" in persona.when_to_use
    assert "## Who Paul Graham Is" in persona.system_prompt
    assert persona.kind == "nuwa"
    assert persona.source_path is not None


def test_load_persona_skill_falls_back_to_stem(tmp_path: Path) -> None:
    path = tmp_path / "naval.md"
    path.write_text("# Naval\n\nFocus on leverage.\n", encoding="utf-8")

    persona = load_persona_skill(path)
    assert persona.name == "naval"
    assert "leverage" in persona.system_prompt


def test_scan_persona_skills_discovers_multiple_files(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("---\nname: alpha\n---\n# Alpha\n", encoding="utf-8")
    (tmp_path / "b.md").write_text("---\nname: beta\n---\n# Beta\n", encoding="utf-8")

    found = scan_persona_skills(tmp_path)
    assert set(found.keys()) == {"alpha", "beta"}


def test_persona_manager_loads_skill_personas(tmp_path: Path) -> None:
    skill_dir = tmp_path / "personas"
    skill_dir.mkdir()
    (skill_dir / "pg.md").write_text(SAMPLE_SKILL, encoding="utf-8")

    mgr = PersonaManager(
        personas_path=tmp_path / "personas.json", skill_dirs=[skill_dir]
    )
    assert "paul-graham" in mgr.list()
    p = mgr.get("paul-graham")
    assert p.kind == "nuwa"
    assert "Paul Graham" in p.system_prompt


def test_persona_manager_import_skill(tmp_path: Path) -> None:
    source = tmp_path / "external.md"
    source.write_text(SAMPLE_SKILL, encoding="utf-8")
    skill_dir = tmp_path / "imported"

    mgr = PersonaManager(
        personas_path=tmp_path / "personas.json", skill_dirs=[skill_dir]
    )
    persona = mgr.import_skill(source, dest_dir=skill_dir)
    assert persona.name == "paul-graham"
    assert (skill_dir / "paul-graham.md").exists()

    # Re-loading the manager should pick up the imported skill.
    mgr2 = PersonaManager(
        personas_path=tmp_path / "personas.json", skill_dirs=[skill_dir]
    )
    assert "paul-graham" in mgr2.list()


def test_persona_manager_match_for_query(tmp_path: Path) -> None:
    skill_dir = tmp_path / "personas"
    skill_dir.mkdir()
    (skill_dir / "pg.md").write_text(SAMPLE_SKILL, encoding="utf-8")

    mgr = PersonaManager(
        personas_path=tmp_path / "personas.json", skill_dirs=[skill_dir]
    )
    matches = mgr.match_for_query("help me evaluate a startup idea")
    assert len(matches) >= 1
    assert matches[0].name == "paul-graham"
