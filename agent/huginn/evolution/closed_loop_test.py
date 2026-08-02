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


def main() -> int:
    tests = [
        ("failure -> rule -> application", test_failure_to_rule_to_application),
        ("success -> skill -> retrieval", test_success_to_skill_to_retrieval),
        ("principle quality gate", test_principle_quality_gate),
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
