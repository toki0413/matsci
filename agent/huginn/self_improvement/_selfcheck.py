"""Self-check for ClawBench metrics in self_improvement/core.py. No frameworks, just asserts.

Run: python -m huginn.self_improvement._selfcheck
"""

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from huginn.self_improvement.core import (
    BenchmarkCase,
    BenchmarkResult,
    BenchmarkSuite,
    MultiTrialResult,
    rubric_evaluator,
)


def _r(case_id, success, score, duration_ms=10.0):
    return BenchmarkResult(
        case_id=case_id, task="test", success=success, score=score,
        response="ok", duration_ms=duration_ms,
    )


def test_finalscore_perfect():
    suite = BenchmarkSuite("t")
    suite.add(BenchmarkCase(task="t1", category="c1"))
    suite.add(BenchmarkCase(task="t2", category="c2"))
    runs = [[_r("t1", True, 1.0), _r("t2", True, 1.0)]] * 3
    mt = suite._compile_multi_trial(runs, 3)
    assert mt.avg_score == 1.0
    assert mt.pass_all_rate == 1.0
    assert mt.pass_any_rate == 1.0
    assert mt.final_score == 100.0
    assert mt.coverage == 1.0


def test_finalscore_all_fail():
    suite = BenchmarkSuite("t")
    suite.add(BenchmarkCase(task="t1", category="c1"))
    runs = [[_r("t1", False, 0.0)]] * 3
    mt = suite._compile_multi_trial(runs, 3)
    assert mt.final_score == 0.0
    assert mt.coverage == 0.0


def test_finalscore_mixed():
    suite = BenchmarkSuite("t")
    suite.add(BenchmarkCase(task="t1", category="c1"))
    suite.add(BenchmarkCase(task="t2", category="c2"))
    runs = [[_r("t1", True, 0.8), _r("t2", False, 0.0)]] * 3
    mt = suite._compile_multi_trial(runs, 3)
    assert abs(mt.avg_score - 0.4) < 1e-6
    assert mt.pass_all_rate == 0.5
    assert mt.pass_any_rate == 0.5
    r_all = 0.5 ** (1.0 / 3.0)
    r_any = 1.0 - (1.0 - 0.5) ** (1.0 / 3.0)
    expected = round(100.0 * (0.4 ** 0.40) * (r_all ** 0.45) * (r_any ** 0.15), 2)
    assert abs(mt.final_score - expected) < 0.01
    assert mt.coverage == 0.5


def test_rubric_normalization():
    suite = BenchmarkSuite("t")
    suite.add(BenchmarkCase(
        task="t1", category="c1",
        rubric_items=[{"criterion": "x", "weight": 1, "keywords": ["hello"]}],
        evaluator=rubric_evaluator,
    ))
    runs = [[_r("t1", True, 100.0)]] * 3
    mt = suite._compile_multi_trial(runs, 3)
    assert mt.avg_score == 1.0  # 100/100 normalized to 1.0
    assert mt.final_score == 100.0


def test_matsci_cases():
    suite = BenchmarkSuite("ms")
    suite.materials_science_research_cases()
    assert len(suite.cases) >= 12
    cats = {c.category for c in suite.cases}
    required = {
        "structure", "electronic", "database", "symbolic", "literature",
        "multiscale", "phase_diagram", "thermodynamics", "degradation",
        "research_design", "mechanical", "inverse_design",
    }
    assert required.issubset(cats)


def test_inverse_design_cases():
    """D: 反向推理 case 存在且 rubric 结构正确."""
    suite = BenchmarkSuite("ms")
    suite.materials_science_research_cases()
    inv_cases = [c for c in suite.cases if c.category == "inverse_design"]
    assert len(inv_cases) == 2
    for c in inv_cases:
        assert "reverse_reasoning" in c.tags
        assert c.rubric_items  # 必须有 rubric
        assert len(c.rubric_items) >= 3  # 至少 3 条评分标准
        # 每个 rubric item 必须有 criterion + weight + keywords
        for item in c.rubric_items:
            assert "criterion" in item
            assert "weight" in item
            assert "keywords" in item
            assert item["weight"] > 0


def test_summary_multi_trial():
    suite = BenchmarkSuite("t")
    suite.add(BenchmarkCase(task="t1", category="c1"))
    suite.add(BenchmarkCase(task="t2", category="c2"))
    runs = [[_r("t1", True, 1.0), _r("t2", True, 1.0)]] * 3
    mt = suite._compile_multi_trial(runs, 3)
    s = suite.summary(mt)
    assert s["final_score"] == 100.0
    assert "case_results" in s
    assert len(s["case_results"]) == 2


def test_summary_plain_list():
    suite = BenchmarkSuite("t")
    suite.add(BenchmarkCase(task="t1", category="c1"))
    s = suite.summary([_r("t1", True, 1.0)])
    assert s["total"] == 1
    assert s["passed"] == 1


def test_cost_field():
    r = BenchmarkResult("id", "task", True, 1.0, "resp", 10.0)
    assert r.cost == 0.0
    r2 = BenchmarkResult("id", "task", True, 1.0, "resp", 10.0, cost=0.05)
    assert r2.cost == 0.05


class _FakeAgent:
    def __init__(self):
        self.calls = 0

    async def chat(self, task, thread_id=""):
        self.calls += 1
        yield {"messages": [{"content": "diamond cubic fd-3m"}]}


async def test_checkpoint_resume():
    suite = BenchmarkSuite("cp")
    suite.add(BenchmarkCase(
        task="silicon structure",
        expected_keywords=["diamond", "cubic", "fd-3m"],
        category="structure",
    ))
    agent = _FakeAgent()
    ckpt = tempfile.mktemp(suffix=".json")
    await suite.run_multi_trial(agent, trials=2, checkpoint_path=ckpt)
    assert agent.calls == 2
    agent2 = _FakeAgent()
    await suite.run_multi_trial(agent2, trials=3, checkpoint_path=ckpt)
    assert agent2.calls == 1  # only the 3rd trial, first 2 resumed from disk
    Path(ckpt).unlink(missing_ok=True)


if __name__ == "__main__":
    test_finalscore_perfect()
    print("[OK] FinalScore perfect = 100")
    test_finalscore_all_fail()
    print("[OK] FinalScore all-fail = 0")
    test_finalscore_mixed()
    print("[OK] FinalScore mixed formula")
    test_rubric_normalization()
    print("[OK] Rubric score normalization")
    test_matsci_cases()
    print("[OK] materials_science_research_cases: 13 cases, 12 categories (incl. inverse_design)")
    test_inverse_design_cases()
    print("[OK] inverse_design cases: 2 reverse-reasoning tasks with rubric")
    test_summary_multi_trial()
    print("[OK] summary(MultiTrialResult)")
    test_summary_plain_list()
    print("[OK] summary(list) backward compat")
    test_cost_field()
    print("[OK] BenchmarkResult.cost")
    asyncio.run(test_checkpoint_resume())
    print("[OK] Checkpoint resume")
    print("\nAll self-checks passed.")
