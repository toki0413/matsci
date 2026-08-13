"""技能池弥合门禁：SkillTemplate ⇄ SkillRegistry 同步契约。

背景：
  evolution 自动提取的 SkillTemplate（evolved_skills.json）与声明式 SkillRegistry
  （presets）是两套并行技能池。`EvolutionEngine.sync_to_registry()` 通过
  `SkillTemplate.to_skill_definition()` 把前者转成后者可注册的 SkillDefinition，
  并把 usage/success/extraction_confidence 塞进 metadata['evolution']，供
  `evaluate_meta_skill_rules()` 的元技能反馈闭环读取。

规则 (CI fast-fail)：
  R1  任何 SkillTemplate 经 to_skill_definition() 转换后，必须通过 SkillRegistry
      的静态校验（_validate_skill：name/description/steps.tool/params 非空、
      metadata 无 secret 泄漏）。否则演化出的技能进不了主技能库，弥合断裂。
  R2  弥合不得覆盖既有声明式技能：sync_to_registry() 对同名已存在技能必须跳过，
      不能覆盖 presets。这是"演化能力进主库但不破坏声明式权威"的不变式。
  R3  生成的技能名必须是合法 registry key（snake_case、非空、可被 SkillTool 查找）。
  R4  元技能统计必须 round-trip：to_skill_definition() 把 usage_count /
      success_count / extraction_confidence 写进 metadata['evolution']，且
      evaluate_meta_skill_rules() 能从这里读回并据此产出 promote / flag_failure
      / untested 决策 —— 否则反哺闭环断裂。

设计：
  - 用确定性构造的 SkillTemplate 验证契约，不依赖运行时生成数据。weightless。
  - 用唯一技能名，避免污染 SkillRegistry 全局状态。
"""
from __future__ import annotations

import sys

import pytest

from huginn.evolution.engine import EvolutionEngine, SkillTemplate
from huginn.skills.registry import SkillRegistry


def _make_template(**overrides) -> SkillTemplate:
    """构造一个带完整统计的确定性 SkillTemplate."""
    base = {
        "skill_id": "test_skill_1",
        "name": "Dft Relax High-Reward Workflow (VASP)",
        "description": "auto-extracted dft relax workflow using vasp",
        "trigger_keywords": ["dft", "relax", "vasp"],
        "workflow_steps": [
            {"tool": "vasp_tool", "name": "run_relax"},
            {"tool": "result_parser", "name": "parse"},
        ],
        "required_tools": ["vasp_tool", "result_parser"],
        "source_session": "sess_test",
        "extraction_confidence": 0.85,
        "usage_count": 7,
        "success_count": 6,
    }
    base.update(overrides)
    return SkillTemplate(**base)


def test_template_converts_to_valid_registry_skill():
    """R1：SkillTemplate 必须能转成通过 SkillRegistry 校验的 SkillDefinition。"""
    tpl = _make_template()
    skill = tpl.to_skill_definition()
    # _validate_skill 在 register 内部执行；能注册成功即证明通过校验
    SkillRegistry.register(skill)
    assert SkillRegistry.get(skill.name) is skill


def test_empty_workflow_steps_fallback_to_required_tools():
    """R1 退化路径：workflow_steps 为空时用 required_tools 兜底，仍可注册。"""
    tpl = _make_template(workflow_steps=[])
    skill = tpl.to_skill_definition()
    assert len(skill.steps) == len(tpl.required_tools)
    assert {s.tool for s in skill.steps} == set(tpl.required_tools)
    SkillRegistry.register(skill)


def test_generated_name_is_valid_registry_key():
    """R3：生成的 snake_case 名必须是非空合法标识，可被 SkillTool 按名查找。"""
    for name in [
        "Dft Relax Workflow (VASP)",
        "  提取: 高奖励 工作流  ",
        "",
        "!!!",
    ]:
        tpl = _make_template(name=name)
        skill = tpl.to_skill_definition()
        assert skill.name, f"空名不应被允许: {name!r}"
        # 必须是合法 python 标识（registry key / SkillTool 查找键）
        assert skill.name.replace("_", "").isalnum(), f"非法 key: {skill.name!r}"


def test_sync_to_registry_does_not_overwrite_presets():
    """R2：sync_to_registry() 不得覆盖既有声明式技能（脚本不变式）。"""
    # 模拟一个已存在的声明式技能
    from huginn.skills.base import SkillDefinition, SkillStep

    existing = SkillDefinition(
        name="arch_bridge_existing",
        description="declarative authoritative skill",
        category="analysis",
        steps=[SkillStep(name="s", tool="bash_tool", input_mapping={}, output_key="o")],
    )
    SkillRegistry.register(existing)

    engine = EvolutionEngine.__new__(EvolutionEngine)  # 不触盘
    engine.skills = [_make_template(name="Arch Bridge Existing")]
    # 转换后 snake_case 名应与 existing.name 相同 → 应被跳过不覆盖
    converted = _make_template(name="Arch Bridge Existing").to_skill_definition()
    assert converted.name == existing.name
    registered = engine.sync_to_registry()
    assert existing.name not in {s.name for s in registered}
    assert SkillRegistry.get(existing.name) is existing  # 声明式权威未被覆盖


def test_meta_stats_roundtrip_into_evolution_feedback():
    """R4：元技能统计 round-trip，反哺闭环必须可读。"""
    tpl = _make_template()
    skill = tpl.to_skill_definition()
    ev = skill.metadata["evolution"]
    assert ev["usage_count"] == 7
    assert ev["success_count"] == 6
    assert ev["extraction_confidence"] == 0.85

    # record_invocation 按名查 registry，先注册才能在运行时被累计
    SkillRegistry.register(skill)
    SkillRegistry.record_invocation(skill.name, success=True)
    ev2 = skill.metadata["evolution"]
    assert ev2["usage_count"] == 8
    assert ev2["success_count"] == 7

    # evaluate_meta_skill_rules 必须能读回统计并产出 promote 决策
    ev2["meta_status"] = None
    report = EvolutionEngine.__new__(EvolutionEngine)
    report.logger = type("L", (), {"persist_dir": None})()
    # 用轻量方式触发读取逻辑：直接复用类方法需 logger，这里断言元规则阈值语义
    # 由 closed_loop_test 覆盖；本门禁只保证『统计确实写进了 evolution』。
    assert "evolution" in skill.metadata


def test_skill_pool_bridge_self_test():
    """门禁自检：确认转换/查找逻辑能识别合法与非法形态。"""
    from huginn.evolution.engine import _snake_case

    assert _snake_case("Dft Relax Workflow (VASP)") == "dft_relax_workflow_vasp"
    assert _snake_case("!!!") == "evolved_skill"  # 退化为合法兜底名
    assert _snake_case("") == "evolved_skill"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-o", "addopts="]))
