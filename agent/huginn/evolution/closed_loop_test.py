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


def main() -> int:
    tests = [
        ("failure -> rule -> application", test_failure_to_rule_to_application),
        ("success -> skill -> retrieval", test_success_to_skill_to_retrieval),
        ("pattern generalization", test_pattern_generalization),
        ("principle quality gate", test_principle_quality_gate),
        ("usage_count persistence", test_usage_count_persistence),
        ("kwargs generalization", test_kwargs_generalization),
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
