"""Benchmark task definitions and results."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BenchmarkTask:
    """A single benchmark task."""

    id: str
    category: str
    prompt: str
    evaluator: Callable[[str], tuple[bool, str] | tuple[bool, str, float]]
    timeout_seconds: float = 180.0
    tags: list[str] = field(default_factory=list)
    requires_api_key: bool = True
    # 参考答案, 给 LLM judge 用 (regex 评分低时触发 judge 二次评审)
    reference: str | None = None
    # 是否代码题 (judge 会评估 code_quality 维度)
    is_code_task: bool = False

    def evaluate(self, output: str) -> TaskResult:
        """Run the evaluator and return a scored result.

        evaluator 可返回 2 元组 (passed, reason) 或 3 元组 (passed, reason, score),
        2 元组时 score 留 None, 向后兼容旧 evaluator。
        """
        start = time.time()
        result = self.evaluator(output)
        elapsed = time.time() - start
        if len(result) == 3:
            passed, reason, score = result
        else:
            passed, reason = result
            score = None
        return TaskResult(
            task_id=self.id,
            category=self.category,
            passed=passed,
            reason=reason,
            output=output,
            eval_time_seconds=elapsed,
            score=score,
        )


@dataclass
class TaskResult:
    """Outcome of running one benchmark task."""

    task_id: str
    category: str
    passed: bool
    reason: str
    output: str
    exec_time_seconds: float = 0.0
    eval_time_seconds: float = 0.0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    # 数值奖励通道: evaluator 返回 3 元组时填充, 2 元组时留 None
    score: float | None = None


def contains_any(keywords: list[str]) -> Callable[[str], tuple[bool, str]]:
    """Pass if the output contains any of the given keywords."""

    def evaluate(output: str) -> tuple[bool, str]:
        lower = output.lower()
        found = [kw for kw in keywords if kw.lower() in lower]
        if found:
            return True, f"found keywords: {found}"
        return False, f"missing any of: {keywords}"

    return evaluate


def contains_all(keywords: list[str]) -> Callable[[str], tuple[bool, str]]:
    """Pass if the output contains all of the given keywords."""

    def evaluate(output: str) -> tuple[bool, str]:
        lower = output.lower()
        missing = [kw for kw in keywords if kw.lower() not in lower]
        if not missing:
            return True, "all keywords present"
        return False, f"missing keywords: {missing}"

    return evaluate


def matches_regex(pattern: str) -> Callable[[str], tuple[bool, str]]:
    """Pass if the output matches the regex."""
    compiled = re.compile(pattern, re.IGNORECASE)

    def evaluate(output: str) -> tuple[bool, str]:
        if compiled.search(output):
            return True, f"matched pattern: {pattern}"
        return False, f"did not match pattern: {pattern}"

    return evaluate


def exact_match(expected: str) -> Callable[[str], tuple[bool, str]]:
    """Pass if the output exactly equals the expected string (ignoring whitespace)."""

    def evaluate(output: str) -> tuple[bool, str]:
        if output.strip() == expected.strip():
            return True, "exact match"
        return False, f"expected {expected!r}, got {output!r}"

    return evaluate


def python_runs(predicate: Callable[[Any], bool]) -> Callable[[str], tuple[bool, str]]:
    """Pass if the output is valid Python and predicate(code) is True."""

    def evaluate(output: str) -> tuple[bool, str]:
        try:
            code = output.strip()
            if code.startswith("```"):
                lines = code.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                code = "\n".join(lines).strip()
            compile(code, "<bench>", "exec")
            result = predicate(code)
            if result:
                return True, "python code compiles and predicate passes"
            return False, "python code compiles but predicate fails"
        except SyntaxError as e:
            return False, f"syntax error: {e}"
        except Exception as e:
            return False, f"predicate error: {e}"

    return evaluate
