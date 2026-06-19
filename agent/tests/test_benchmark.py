"""Tests for benchmark and self-improvement loop."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from huginn.agent import HuginnAgent
from huginn.benchmark import (
    BenchmarkCase,
    BenchmarkSuite,
    keyword_evaluator,
    numeric_evaluator,
)
from huginn.benchmark.core import SelfImprovementLoop, llm_judge_evaluator
from huginn.memory.longterm import LongTermMemory
from huginn.memory.manager import MemoryManager


class _FakeAgent:
    """Agent that echoes predefined responses."""

    def __init__(self, responses: dict[str, str]) -> None:
        self.responses = responses

    async def chat(self, task: str, thread_id: str = "default"):
        response = self.responses.get(task, "")
        yield {"messages": [type("Msg", (), {"content": response})()]}


class TestBenchmarkSuite:
    def test_keyword_evaluator_pass(self):
        case = BenchmarkCase(
            task="what is Si structure?",
            expected_keywords=["diamond", "cubic"],
        )
        success, score = keyword_evaluator("Silicon has a diamond cubic lattice", case)
        assert success is True
        assert score == 1.0

    def test_keyword_evaluator_partial(self):
        case = BenchmarkCase(task="band gap?", expected_keywords=["1.1", "eV"])
        success, score = keyword_evaluator("The band gap is 1.1", case)
        assert success is False
        assert score == 0.5

    def test_suite_summary(self):
        suite = BenchmarkSuite()
        suite.add(
            BenchmarkCase(task="pass", expected_keywords=["ok"]),
        )
        results = [
            type("R", (), {"success": True, "score": 1.0, "duration_ms": 10})(),
            type("R", (), {"success": False, "score": 0.0, "duration_ms": 20})(),
        ]
        summary = suite.summary(results)
        assert summary["total"] == 2
        assert summary["passed"] == 1
        assert summary["failed"] == 1
        assert summary["avg_score"] == 0.5

    @pytest.mark.asyncio
    async def test_suite_run(self):
        suite = BenchmarkSuite()
        suite.add(BenchmarkCase(task="t1", expected_keywords=["yes"]))
        agent = _FakeAgent({"t1": "yes indeed"})
        results = await suite.run(agent)
        assert len(results) == 1
        assert results[0].success is True

    def test_numeric_evaluator_pass(self):
        case = BenchmarkCase(
            task="band gap",
            expected_value=1.1,
            tolerance=0.05,
            evaluator=numeric_evaluator,
        )
        success, score = numeric_evaluator("The band gap is 1.12 eV", case)
        assert success is True
        assert score > 0.75

    def test_numeric_evaluator_fail(self):
        case = BenchmarkCase(
            task="band gap",
            expected_value=1.1,
            tolerance=0.05,
            evaluator=numeric_evaluator,
        )
        success, score = numeric_evaluator("The band gap is 2.0 eV", case)
        assert success is False

    def test_llm_judge_evaluator(self):
        def fake_judge(prompt: str) -> str:
            return '{"score": 0.9, "reason": "correct"}'

        case = BenchmarkCase(task="t", rubric="correct")
        evaluator = llm_judge_evaluator(fake_judge)
        success, score = evaluator("answer", case)
        assert success is True
        assert score == 0.9


class TestSelfImprovementLoop:
    @pytest.mark.asyncio
    async def test_failures_stored_in_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            longterm = LongTermMemory(db_path=Path(tmp) / "memory.db")
            memory = MemoryManager(longterm=longterm)

            suite = BenchmarkSuite()
            suite.add(BenchmarkCase(task="t1", expected_keywords=["missing"]))
            loop = SelfImprovementLoop(suite=suite, memory_manager=memory)

            agent = _FakeAgent({"t1": "wrong answer"})
            report = await loop.evaluate(agent)
            assert report["summary"]["failed"] == 1

            failures = longterm.list_by_category("benchmark_failure")
            assert len(failures) >= 1


class TestHuginnAgentBenchmark:
    @pytest.mark.asyncio
    async def test_agent_run_benchmark(self):
        with tempfile.TemporaryDirectory() as tmp:
            longterm = LongTermMemory(db_path=Path(tmp) / "memory.db")
            agent = HuginnAgent(memory_manager=MemoryManager(longterm=longterm))

            suite = BenchmarkSuite()
            suite.add(BenchmarkCase(task="band gap", expected_keywords=["1.1"]))

            fake = _FakeAgent({"band gap": "The band gap is 1.1 eV"})
            agent.chat = fake.chat  # type: ignore[method-assign]
            report = await agent.run_benchmark(suite=suite, store_failures=False)
            assert report["summary"]["passed"] == 1
