"""Lightweight benchmark framework for HuginnAgent.

A benchmark case is a task plus an evaluator. The suite runs each case
against an agent, scores the result, and the self-improvement loop stores
failures in long-term memory so the agent can learn over time.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from huginn.memory.manager import MemoryManager

EvaluatorT = Callable[[str, "BenchmarkCase"], tuple[bool, float]]
"""Evaluator(response, case) -> (success, score)."""


def keyword_evaluator(response: str, case: BenchmarkCase) -> tuple[bool, float]:
    """Pass if all expected keywords are present (case-insensitive)."""
    text = response.lower()
    matches = sum(1 for kw in case.expected_keywords if kw.lower() in text)
    if not case.expected_keywords:
        return True, 1.0
    success = matches == len(case.expected_keywords)
    score = matches / len(case.expected_keywords)
    return success, round(score, 2)


def numeric_evaluator(response: str, case: BenchmarkCase) -> tuple[bool, float]:
    """Pass if a number near ``expected_value`` appears in the response.

    Tolerance defaults to 1% relative or absolute, whichever is larger.
    """
    import re

    expected = case.expected_value
    if expected is None:
        return keyword_evaluator(response, case)

    numbers = [
        float(m) for m in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", response)
    ]
    if not numbers:
        return False, 0.0

    tolerance = case.tolerance or max(abs(expected) * 0.01, 0.01)
    best = min(abs(n - expected) for n in numbers)
    success = best <= tolerance
    score = max(0.0, 1.0 - best / (tolerance * 2)) if tolerance > 0 else 0.0
    return success, round(score, 2)


def llm_judge_evaluator(
    judge_model: Callable[[str], str],
) -> EvaluatorT:
    """Return an evaluator that asks a small LLM to score the answer 0-1.

    The judge is given the task, rubric, and response, and must reply with a
    single JSON object: {"score": float, "reason": str}.
    """

    def evaluate(response: str, case: BenchmarkCase) -> tuple[bool, float]:
        prompt = (
            "You are a strict but fair grader. Evaluate the answer below for the task.\n\n"
            f"Task: {case.task}\n"
            f"Rubric: {case.rubric or 'Answer should be correct and complete.'}\n\n"
            f"Answer: {response[:2000]}\n\n"
            'Respond ONLY with JSON: {"score": float between 0 and 1, "reason": string}'
        )
        try:
            raw = judge_model(prompt)
            # Extract JSON from possible markdown code block.
            if "```" in raw:
                raw = raw.split("```")[1].strip("json").strip()
            data = json.loads(raw)
            score = float(data.get("score", 0.0))
            return score >= 0.8, round(score, 2)
        except Exception:
            return False, 0.0

    return evaluate


def rubric_evaluator(response: str, case: BenchmarkCase) -> tuple[bool, float]:
    """Score against weighted rubric items (RCBench-style).

    Each rubric item has: criterion (str), weight (float), keywords (list[str]).
    An item is "met" if all its keywords appear in the response (case-insensitive).
    Score = sum(met weights) / sum(total weights) * 100.
    Falls back to keyword_evaluator if rubric_items is empty.
    """
    if not case.rubric_items:
        return keyword_evaluator(response, case)

    text = response.lower()
    total_weight = 0.0
    met_weight = 0.0
    for item in case.rubric_items:
        weight = float(item.get("weight", 1.0))
        keywords = item.get("keywords", [])
        total_weight += weight
        if not keywords:
            # no keywords = criterion met by default (e.g. code execution succeeded)
            met_weight += weight
        elif all(kw.lower() in text for kw in keywords):
            met_weight += weight

    if total_weight == 0:
        return True, 100.0
    score = round(met_weight / total_weight * 100, 1)
    # ponytail: pass threshold at 50 (RCBench's "matches paper" anchor)
    return score >= 50, score


@dataclass
class BenchmarkCase:
    """A single benchmark task."""

    task: str
    expected_keywords: list[str] = field(default_factory=list)
    expected_value: float | None = None
    tolerance: float | None = None
    rubric: str | None = None
    evaluator: EvaluatorT = keyword_evaluator
    category: str = "general"
    tags: list[str] = field(default_factory=list)
    case_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    # RCBench-style weighted rubric items. Each item:
    # {"criterion": str, "weight": float, "keywords": [str], "type": "text"|"image"}
    # When populated, rubric_evaluator scores each criterion by keyword presence
    # and computes a weighted sum scaled to 0-100.
    rubric_items: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    """Result of running one benchmark case."""

    case_id: str
    task: str
    success: bool
    score: float
    response: str
    duration_ms: float
    error: str | None = None


class BenchmarkSuite:
    """Collection of benchmark cases with execution and scoring."""

    def __init__(self, name: str = "default") -> None:
        self.name = name
        self.cases: list[BenchmarkCase] = []

    def add(self, case: BenchmarkCase) -> BenchmarkSuite:
        self.cases.append(case)
        return self

    def add_defaults(self) -> BenchmarkSuite:
        """Register a small set of materials-science sanity checks."""
        self.add(
            BenchmarkCase(
                task="What is the crystal structure of silicon at room temperature?",
                expected_keywords=["diamond", "cubic"],
                category="materials",
                tags=["structure"],
            )
        )
        self.add(
            BenchmarkCase(
                task="Calculate the band gap of silicon in eV.",
                expected_value=1.1,
                tolerance=0.05,
                evaluator=numeric_evaluator,
                category="materials",
                tags=["electronic"],
            )
        )
        self.add(
            BenchmarkCase(
                task="Set up a harmonic oscillator model with mass 1 and k=4.",
                expected_value=2.0,
                tolerance=0.1,
                evaluator=numeric_evaluator,
                category="unified",
                tags=["math"],
            )
        )
        return self

    async def run(
        self,
        agent: Any,
        thread_id: str = "benchmark",
    ) -> list[BenchmarkResult]:
        """Run all cases against ``agent`` and return scored results."""
        results: list[BenchmarkResult] = []
        for case in self.cases:
            start = time.time()
            response = ""
            error: str | None = None
            try:
                response = await self._invoke_agent(agent, case.task, thread_id)
            except Exception as exc:
                error = str(exc)
            duration_ms = round((time.time() - start) * 1000, 2)

            if error:
                results.append(
                    BenchmarkResult(
                        case_id=case.case_id,
                        task=case.task,
                        success=False,
                        score=0.0,
                        response=response,
                        duration_ms=duration_ms,
                        error=error,
                    )
                )
                continue

            success, score = case.evaluator(response, case)
            results.append(
                BenchmarkResult(
                    case_id=case.case_id,
                    task=case.task,
                    success=success,
                    score=score,
                    response=response,
                    duration_ms=duration_ms,
                )
            )
        return results

    @staticmethod
    async def _invoke_agent(agent: Any, task: str, thread_id: str) -> str:
        """Collect the final response from an async agent."""
        final_response = ""
        async for state in agent.chat(task, thread_id=thread_id):
            messages = state.get("messages", [])
            for msg in messages:
                content = getattr(msg, "content", None)
                if content:
                    final_response = str(content)
        return final_response

    def summary(self, results: list[BenchmarkResult]) -> dict[str, Any]:
        """Return aggregate statistics for a result set."""
        if not results:
            return {"total": 0, "passed": 0, "failed": 0, "avg_score": 0.0}
        passed = sum(1 for r in results if r.success)
        return {
            "total": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "avg_score": round(sum(r.score for r in results) / len(results), 3),
            "avg_duration_ms": round(
                sum(r.duration_ms for r in results) / len(results), 2
            ),
        }


class SelfImprovementLoop:
    """Run benchmarks and feed failures back into long-term memory."""

    def __init__(
        self,
        suite: BenchmarkSuite,
        memory_manager: MemoryManager,
    ) -> None:
        self.suite = suite
        self.memory = memory_manager

    async def evaluate(
        self,
        agent: Any,
        store_failures: bool = True,
    ) -> dict[str, Any]:
        """Run the suite and optionally memorize failures for future learning."""
        results = await self.suite.run(agent)
        summary = self.suite.summary(results)

        if store_failures:
            for r in results:
                if not r.success:
                    self.memory.remember(
                        content=(
                            f"Benchmark failure [{r.case_id}]: task='{r.task}' "
                            f"score={r.score} response='{r.response[:500]}'"
                        ),
                        category="benchmark_failure",
                        tags=["benchmark", r.task[:20]],
                        importance=0.7,
                        tier="mid",
                    )

        return {"summary": summary, "results": results}
