"""端到端闭环验证: 规则生成 -> 应用 -> usage_count > 0.

跑法: cd agent && python -m huginn.evolution.closed_loop_test

三个场景跑通就说明 Step 1-7 的修复把闭环接上了:
  1. 失败 -> evolve_from_failures -> apply_heuristic_fix -> usage_count > 0
  2. 成功 -> evolve_from_successes -> get_relevant_skills 命中
  3. stable_principle 质量门槛拒绝噪声

ponytail: 不用 pytest fixture, 直接 mkdtemp + assert, 一个文件搞定.
"""
from __future__ import annotations

import shutil
import sys
import tempfile

from huginn.evolution.engine import EvolutionEngine
from huginn.evolution.logger import ExecutionLogger
from huginn.memory.longterm import _validate_principle


def _fresh_logger() -> tuple[ExecutionLogger, str]:
    """每个测试用独立临时目录, 不污染全局 ~/.huginn/logs."""
    tmp = tempfile.mkdtemp(prefix="huginn_cl_")
    return ExecutionLogger(persist_dir=tmp), tmp


def test_failure_to_rule_to_application() -> None:
    """失败两次 -> evolve 产出规则 -> 同样错误再发生命中规则."""
    logger, tmp = _fresh_logger()
    try:
        err = "File '/tmp/missing.cif' not found"
        tool_input = {"file_path": "/tmp/missing.cif"}
        # 两次同样的失败, 凑够 get_failure_patterns 的 min_count=2
        for _ in range(2):
            logger.log_tool_call(
                session_id="s1",
                tool_name="read_file",
                tool_input=tool_input,
                error=err,
            )

        engine = EvolutionEngine(logger=logger)
        new_rules = engine.evolve_from_failures()
        assert len(new_rules) == 1, f"应产出 1 条规则, got {len(new_rules)}"
        rule = new_rules[0]
        assert rule.rule_type == "heuristic_fix"

        # 同样的错误再发生, 看规则能不能命中
        fix = engine.apply_heuristic_fix("read_file", tool_input, err)
        assert fix is not None, "规则应命中"
        assert "description" in fix, f"fix 必须含 description, got {fix}"
        assert rule.usage_count > 0, f"usage_count 应 > 0, got {rule.usage_count}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_success_to_skill_to_retrieval() -> None:
    """成功三次 -> evolve 产出 skill -> get_relevant_skills 命中."""
    logger, tmp = _fresh_logger()
    try:
        tool_input = {"file_path": "/tmp/test.cif"}
        # evolve_from_successes 要 >=3 条同类成功记录才提取 skill
        for _ in range(3):
            logger.log_tool_call(
                session_id="s1",
                tool_name="read_file",
                tool_input=tool_input,
                result={"status": "ok", "structure": "parsed"},
                software="VASP",
                calculation_type="relax",
            )

        engine = EvolutionEngine(logger=logger)
        new_skills = engine.evolve_from_successes()
        assert len(new_skills) == 1, f"应产出 1 个 skill, got {len(new_skills)}"
        skill = new_skills[0]

        # query 里带 trigger_keywords, 看能不能检索到
        hits = engine.get_relevant_skills("run a VASP relax calculation")
        assert len(hits) >= 1, "应检索到 skill"
        assert hits[0].skill_id == skill.skill_id
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_evolved_skill_sync_to_registry() -> None:
    """弥合两个技能池: 自动提取的 SkillTemplate 应能同步进 SkillRegistry.

    成功三次 -> evolve_from_successes 产出模板 -> sync_to_registry 注册进
    主技能库 (带 evolution 元数据, 幂等), 且 record_invocation 更新复用统计.
    """
    from huginn.skills.registry import SkillRegistry

    logger, tmp = _fresh_logger()
    try:
        tool_input = {"file_path": "/tmp/test.cif"}
        for _ in range(3):
            logger.log_tool_call(
                session_id="s1",
                tool_name="read_file",
                tool_input=tool_input,
                result={"status": "ok", "structure": "parsed"},
                software="VASP",
                calculation_type="relax",
            )

        engine = EvolutionEngine(logger=logger)
        new_skills = engine.evolve_from_successes()
        assert len(new_skills) == 1

        synced = engine.sync_to_registry()
        assert len(synced) == 1, f"应同步 1 个技能, got {len(synced)}"
        name = synced[0].name
        assert SkillRegistry.get(name) is not None, "应注册进 SkillRegistry"

        # 元数据带演化统计
        ev = SkillRegistry.get(name).metadata.get("evolution", {})
        assert ev.get("skill_id"), "应带 skill_id"
        assert "extraction_confidence" in ev

        # 幂等: 再同步一次不新增
        assert len(engine.sync_to_registry()) == 0

        # record_invocation 更新复用/成败统计
        SkillRegistry.record_invocation(name, True)
        SkillRegistry.record_invocation(name, False)
        ev2 = SkillRegistry.get(name).metadata["evolution"]
        assert ev2["usage_count"] == 2, f"usage_count 应=2, got {ev2['usage_count']}"
        assert ev2["success_count"] == 1

        # 清理, 只移除本次新增技能, 不污染全局 registry 的 presets
        if name in SkillRegistry._skills:
            del SkillRegistry._skills[name]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_pattern_generalization() -> None:
    """两个不同文件路径的同类错误应归到同一条规则, 而非各生成一条."""
    from huginn.evolution.logger import _generalize_error

    e1 = "Error: File '/tmp/missing.cif' not found"
    e2 = "Error: File '/data/other.csv' not found"
    assert _generalize_error(e1) == _generalize_error(e2), (
        "不同路径应泛化到同一模式"
    )

    # 端到端: 两次不同路径的失败只产出 1 条规则, 第三次不同路径也能命中
    logger, tmp = _fresh_logger()
    try:
        for path in ["/tmp/a.cif", "/data/b.csv", "/home/c.txt"]:
            logger.log_tool_call(
                session_id="s1",
                tool_name="read_file",
                tool_input={"file_path": path},
                error=f"Error: File '{path}' not found",
            )
        engine = EvolutionEngine(logger=logger)
        rules = engine.evolve_from_failures()
        assert len(rules) == 1, f"3 条不同路径应只产 1 条规则, got {len(rules)}"
        # 第四个新路径也该命中
        fix = engine.apply_heuristic_fix(
            "read_file", {"file_path": "/new/x.pkl"},
            "Error: File '/new/x.pkl' not found",
        )
        assert fix is not None, "新路径应命中泛化后的规则"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_principle_quality_gate() -> None:
    """质量门槛: 拒同义反复, 拒超长噪声, 放行合理原则."""
    # 同义反复 — 命中 _BAD_PATTERNS
    assert _validate_principle("avoid tool failure: Tool 'ls' encountered an issue.") is False
    # 超长 — 超过 500 字符上限
    assert _validate_principle("x" * 600) is False
    # 合理原则
    assert _validate_principle(
        "verify file existence before read_file when path comes from user input"
    ) is True
    assert _validate_principle(
        "check convergence before proceeding to next calculation step"
    ) is True


def test_usage_count_persistence() -> None:
    """apply_heuristic_fix 命中后 usage_count 应持久化, 重 load 不丢.

    修 engine.py: apply_heuristic_fix 命中后调 _save_rules, 否则跨 session 丢.
    """
    logger, tmp = _fresh_logger()
    try:
        for _ in range(2):
            logger.log_tool_call(
                session_id="s1",
                tool_name="read_file",
                tool_input={"file_path": "/tmp/x.cif"},
                error="File '/tmp/x.cif' not found",
            )
        engine = EvolutionEngine(logger=logger)
        engine.evolve_from_failures()
        # 命中一次, usage_count 应该 1
        fix = engine.apply_heuristic_fix(
            "read_file", {"file_path": "/tmp/x.cif"},
            "File '/tmp/x.cif' not found",
        )
        assert fix is not None, "规则应命中"
        assert engine.rules[0].usage_count == 1, (
            f"usage_count 应 1, got {engine.rules[0].usage_count}"
        )
        # 重新 load, usage_count 不丢 (验证 _save_rules 生效)
        engine2 = EvolutionEngine(logger=logger)
        assert engine2.rules[0].usage_count == 1, (
            f"重 load 后 usage_count 应 1, got {engine2.rules[0].usage_count}"
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_kwargs_generalization() -> None:
    """不同 content/command 值的 kwargs 错误应归到同一条规则.

    修 logger.py: _generalize_error 抽象 with kwargs {...} 块.
    不修的话 file_write_tool 每个不同 content = 独立规则, bash_tool 每个不同 command = 独立规则.
    """
    from huginn.evolution.logger import _generalize_error

    # file_write_tool: 不同 content 值应泛化到同一模式
    e1 = "Error invoking tool 'file_write_tool' with kwargs {'content': '#!/usr/bin/python3\nprint(\"a\")'} with error: invalid"
    e2 = "Error invoking tool 'file_write_tool' with kwargs {'content': 'import numpy as np\nprint(1)'} with error: invalid"
    assert _generalize_error(e1) == _generalize_error(e2), (
        f"不同 content 应泛化到同一模式\n  e1={_generalize_error(e1)}\n  e2={_generalize_error(e2)}"
    )

    # bash_tool: 不同 command 值应泛化到同一模式
    e3 = 'Error invoking tool \'bash_tool\' with kwargs {\'command\': \'["ls", "-la"]\'} with error:\n command: Input should be a valid list'
    e4 = 'Error invoking tool \'bash_tool\' with kwargs {\'command\': \'["pwd"]\'} with error:\n command: Input should be a valid list'
    assert _generalize_error(e3) == _generalize_error(e4), (
        f"不同 command 应泛化到同一模式\n  e3={_generalize_error(e3)}\n  e4={_generalize_error(e4)}"
    )

    # 端到端: 3 个不同 content 的失败只产 1 条规则
    logger, tmp = _fresh_logger()
    try:
        for content in ["print('a')", "import os", "x = 1"]:
            logger.log_tool_call(
                session_id="s1",
                tool_name="file_write_tool",
                tool_input={"content": content},
                error=f"Error invoking tool 'file_write_tool' with kwargs {{'content': '{content}'}} with error: invalid",
            )
        engine = EvolutionEngine(logger=logger)
        rules = engine.evolve_from_failures()
        assert len(rules) == 1, f"3 条不同 content 应只产 1 条规则, got {len(rules)}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_meta_skill_rules_feedback() -> None:
    """元技能规则反哺: record_invocation 统计 → promote / flag_failure / untested.

    复用高+成功率高 → promote; 调用够多但成功率低 → flag_failure;
    从未被调 → untested (信息性, 不删). 并用 record_invocation 走真数据路径.
    """
    from huginn.skills.base import SkillDefinition, SkillStep
    from huginn.skills.registry import SkillRegistry

    def _mk(name: str) -> None:
        skill = SkillDefinition(
            name=name,
            description="meta test skill",
            category="distilled",
            steps=[SkillStep(name="s1", tool="read_file", input_mapping={},
                             output_key="o", on_failure="skip")],
            required_tools=["read_file"],
            tags=["test"],
            metadata={"evolution": {"usage_count": 0, "success_count": 0}},
        )
        SkillRegistry.register(skill)

    # 用 record_invocation 的真实现累积统计 (闭环数据输入)
    _mk("meta_promote_candidate")
    for _ in range(5):
        SkillRegistry.record_invocation("meta_promote_candidate", True)

    _mk("meta_flag_failure_candidate")
    SkillRegistry.record_invocation("meta_flag_failure_candidate", True)
    for _ in range(2):
        SkillRegistry.record_invocation("meta_flag_failure_candidate", False)

    _mk("meta_untested_candidate")  # 从未 record_invocation → usage=0

    try:
        logger, tmp = _fresh_logger()
        engine = EvolutionEngine(logger=logger)
        report = engine.evaluate_meta_skill_rules()

        promote_names = {r["name"] for r in report["promote_to_primitive"]}
        flag_names = {r["name"] for r in report["flag_high_failure"]}
        assert "meta_promote_candidate" in promote_names, (
            f"5 次调用全成功应 promote, got promote={promote_names}"
        )
        assert "meta_flag_failure_candidate" in flag_names, (
            f"3 次调用 2 失败应 flag_failure, got flag={flag_names}"
        )
        assert "meta_untested_candidate" in report["untested"]

        # 状态已回流到 metadata (非破坏性标记)
        assert (
            SkillRegistry.get("meta_promote_candidate")
            .metadata["evolution"]["meta_status"]
            == "promote"
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        for name in (
            "meta_promote_candidate",
            "meta_flag_failure_candidate",
            "meta_untested_candidate",
        ):
            if name in SkillRegistry._skills:
                del SkillRegistry._skills[name]


def main() -> int:
    tests = [
        ("failure -> rule -> application", test_failure_to_rule_to_application),
        ("success -> skill -> retrieval", test_success_to_skill_to_retrieval),
        ("pattern generalization", test_pattern_generalization),
        ("principle quality gate", test_principle_quality_gate),
        ("usage_count persistence", test_usage_count_persistence),
        ("kwargs generalization", test_kwargs_generalization),
        ("meta skill rules feedback", test_meta_skill_rules_feedback),
    ]
    passed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS: {name}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {name}: {e}")
        except Exception as e:
            print(f"  ERROR: {name}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
