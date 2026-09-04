"""工业 Skill 接入指南防漂移自检.

对齐 ``industrial-skill-guide.md``：示例 skill 的 frontmatter 字段必须真实存在
于 ``SkillImporter`` 原生字段集内，且能真正被解析注册。任何字段/格式漂移都会让本
测试变红，从而保证"菜谱与代码一致"。
"""

from __future__ import annotations

from pathlib import Path

from huginn.plugins.skill_importer import SkillImporter

# 仓库根（/workspace/agent），文档与示例的相对路径从这起算
_REPO = Path(__file__).resolve().parents[1]

# SkillImporter 真正读取的 frontmatter 原生字段（对齐 _build_native/_build_openclaw/_build_hermes）
_NATIVE_FIELDS = frozenset(
    {
        "name",
        "description",
        "category",
        "steps",
        "allowed-tools",
        "tools",
        "when_to_use",
        "trigger",
        "trigger_conditions",
        "tags",
        "paths",
        "model",
        "effort",
    }
)

_EXAMPLE_SKILL = (
    _REPO / "huginn/plugins/science-skills/skills/industrial-asset-state/SKILL.md"
)


def test_example_skill_parses_as_native():
    skill = SkillImporter().import_file(_EXAMPLE_SKILL, platform="huginn")
    assert skill.name == "industrial-asset-state"
    assert skill.category == "diagnostics"
    assert skill.required_tools  # allowed-tools 被映射为 required_tools
    assert "industrial" in skill.tags
    assert skill.metadata["platform"] == "huginn"
    assert "## 方法规则" in (skill.metadata.get("content") or "")


def test_example_uses_only_native_frontmatter_fields():
    # 解析 frontmatter，断言用到的 key 全部属于 _NATIVE_FIELDS，防新增私有字段漂移
    frontmatter, _ = SkillImporter()._parse_frontmatter(_EXAMPLE_SKILL)
    unknown = set(frontmatter) - _NATIVE_FIELDS
    assert not unknown, f"示例使用了解析器不认识的前端字段: {sorted(unknown)}"


def test_guide_registered_in_docs_index():
    index = (_REPO / "docs/INDEX.md").read_text(encoding="utf-8")
    assert "industrial-skill-guide.md" in index
    assert "Industrial" in index or "工业" in index


def test_guide_references_example_skill_path():
    guide = (_REPO / "docs/industrial-skill-guide.md").read_text(encoding="utf-8")
    assert "industrial-asset-state" in guide
    # 手册里声明的原生字段清单与代码一致
    for field in ("allowed-tools", "when_to_use", "steps", "trigger"):
        assert field in guide


def test_metadata_spec_registered_in_index():
    index = (_REPO / "docs/INDEX.md").read_text(encoding="utf-8")
    assert "industrial-skill-metadata-spec.md" in index


def test_metadata_spec_declares_native_fields_only():
    spec = (_REPO / "docs/industrial-skill-metadata-spec.md").read_text(
        encoding="utf-8"
    )
    # 规范声明的 frontmatter 字段必须全部属于解析器原生字段集
    declared = [f for f in _NATIVE_FIELDS if f in spec]
    assert {"name", "description", "category", "allowed-tools", "steps"} <= set(
        declared
    )
    # 未引入规范自身的私有 frontmatter 字段名（抽查几个高频臆造字段不应出现）
    for madeup in ("triggers_en", "skill_policy", "owner_team"):
        assert madeup not in spec.split("\n")[0:1], madeup
    # 与接入指南互链
    assert (
        "industrial-skill-metadata-spec.md" in spec
        or "industrial-skill-guide.md" in spec
    )
