"""MatWorldBench 测试 — 题集完整性 / 评测逻辑 / run_all / BenchGrader 集成.

benchmark 拿 agent 算出来的 dict 跟 expected_result 逐 key 比对,
数值走 tolerance 绝对带, 其余严格相等. 这里覆盖正确/错误/边界/部分分
以及 BenchGrader 包装层.
"""
from __future__ import annotations

import pytest

from huginn.evaluation.matworld_bench import (
    BenchResult,
    BenchTask,
    CATEGORIES,
    MatWorldBench,
)
from huginn.validation.grader import BenchGrader, GraderResult


# ── 题集完整性 ──────────────────────────────────────────────────


def test_ten_tasks_cover_all_categories():
    """内置 10 道题, 五个 category 都覆盖到, id 不重复."""
    bench = MatWorldBench()
    assert len(bench.tasks) == 10

    cats = {t.category for t in bench.tasks}
    assert cats == set(CATEGORIES)

    ids = [t.id for t in bench.tasks]
    assert len(ids) == len(set(ids)), "duplicate task ids"


def test_task_fields_and_tolerance_types():
    """每道题的 tolerance 都是 float dict, expected_result 非空."""
    bench = MatWorldBench()
    for t in bench.tasks:
        assert t.prompt, f"{t.id}: empty prompt"
        assert t.expected_result, f"{t.id}: empty expected_result"
        assert t.tolerance, f"{t.id}: empty tolerance"
        for v in t.tolerance.values():
            assert isinstance(v, float), f"{t.id}: tolerance not float"


# ── evaluate 单题 ───────────────────────────────────────────────


def test_evaluate_correct_answer_passes():
    """Si 带隙给对值 -> passed=True, score=1.0."""
    bench = MatWorldBench()
    res = bench.evaluate("si_bandgap", {"band_gap_eV": 1.12})
    assert isinstance(res, BenchResult)
    assert res.passed is True
    assert res.score == pytest.approx(1.0)
    assert res.details["category"] == "electronic"


def test_evaluate_within_tolerance_passes():
    """落在容差带内 (1.20 vs 1.12, tol=0.15) -> 通过."""
    bench = MatWorldBench()
    res = bench.evaluate("si_bandgap", {"band_gap_eV": 1.20})
    assert res.passed is True
    assert res.score == pytest.approx(1.0)


def test_evaluate_outside_tolerance_fails():
    """超出容差带 (1.50 vs 1.12) -> 不通过, score=0."""
    bench = MatWorldBench()
    res = bench.evaluate("si_bandgap", {"band_gap_eV": 1.50})
    assert res.passed is False
    assert res.score == pytest.approx(0.0)
    # details 里能看到哪个 key 没过
    assert res.details["keys"]["band_gap_eV"]["pass"] is False


def test_evaluate_missing_key_fails():
    """缺 key -> 不通过, reason 标 missing key."""
    bench = MatWorldBench()
    res = bench.evaluate("si_bandgap", {})
    assert res.passed is False
    assert res.score == pytest.approx(0.0)
    assert res.details["keys"]["band_gap_eV"]["reason"] == "missing key"


def test_evaluate_unknown_task():
    """不存在的 task_id -> score=0, details 带 error."""
    bench = MatWorldBench()
    res = bench.evaluate("nope", {"band_gap_eV": 1.0})
    assert res.passed is False
    assert res.score == pytest.approx(0.0)
    assert "unknown" in res.details["error"]


def test_evaluate_partial_score_with_multi_key_task():
    """多 key 的题: 一个对一个错 -> score=0.5, passed=False."""
    task = BenchTask(
        id="multi",
        category="structure",
        prompt="lattice + density",
        expected_result={"a_A": 4.05, "rho": 2.70},
        tolerance={"a_A": 0.05, "rho": 0.10},
    )
    bench = MatWorldBench(tasks=[task])
    # a 对, rho 错
    res = bench.evaluate("multi", {"a_A": 4.06, "rho": 5.0})
    assert res.passed is False
    assert res.score == pytest.approx(0.5)


def test_evaluate_non_numeric_strict_equal():
    """非数值 key 没在 tolerance 里 -> 严格相等."""
    task = BenchTask(
        id="str",
        category="structure",
        prompt="crystal system",
        expected_result={"crystal_system": "cubic"},
        tolerance={},
    )
    bench = MatWorldBench(tasks=[task])
    assert bench.evaluate("str", {"crystal_system": "cubic"}).passed is True
    assert bench.evaluate("str", {"crystal_system": "tetragonal"}).passed is False


# ── run_all ─────────────────────────────────────────────────────


def test_run_all_perfect_evaluator():
    """evaluator 全答对 -> pass_rate=1.0, failed=0."""
    bench = MatWorldBench()

    def evaluator(task: BenchTask):
        return dict(task.expected_result)

    summary = bench.run_all(evaluator)
    assert summary["total"] == 10
    assert summary["passed"] == 10
    assert summary["failed"] == 0
    assert summary["pass_rate"] == pytest.approx(1.0)
    assert len(summary["results"]) == 10


def test_run_all_handles_evaluator_exception():
    """evaluator 抛异常的那道题算 fail, 不影响其余."""
    bench = MatWorldBench()

    def evaluator(task: BenchTask):
        if task.id == "si_bandgap":
            raise RuntimeError("boom")
        return dict(task.expected_result)

    summary = bench.run_all(evaluator)
    assert summary["passed"] == 9
    assert summary["failed"] == 1


# ── BenchGrader 集成 ────────────────────────────────────────────


def test_bench_grader_wraps_bench():
    """BenchGrader 把 BenchResult 折算成 GraderResult."""
    g = BenchGrader()
    assert g.name == "matworld_bench"
    res = g.evaluate({
        "task_id": "fe_bcc_lattice",
        "agent_output": {"lattice_constant_A": 2.866},
    })
    assert isinstance(res, GraderResult)
    assert res.passed is True
    assert res.score == pytest.approx(1.0)
    assert "fe_bcc_lattice" in res.message


def test_bench_grader_missing_task_id():
    """没给 task_id -> score=0, passed=False."""
    g = BenchGrader()
    res = g.evaluate({"agent_output": {"band_gap_eV": 1.12}})
    assert res.passed is False
    assert res.score == pytest.approx(0.0)
    assert "missing" in res.message.lower()


def test_bench_grader_callable():
    """BenchGrader 实例可直接当 callable."""
    g = BenchGrader()
    res = g({"task_id": "si_bandgap", "agent_output": {"band_gap_eV": 1.12}})
    assert res.passed is True
