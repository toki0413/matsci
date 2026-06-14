"""Benchmark task definitions and results."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class BenchmarkTask:
    """A single benchmark task."""

    id: str
    category: str
    prompt: str
    evaluator: Callable[[str], tuple[bool, str]]
    timeout_seconds: float = 120.0
    tags: list[str] = field(default_factory=list)
    requires_api_key: bool = True

    def evaluate(self, output: str) -> TaskResult:
        """Run the evaluator and return a scored result."""
        start = time.time()
        passed, reason = self.evaluator(output)
        elapsed = time.time() - start
        return TaskResult(
            task_id=self.id,
            category=self.category,
            passed=passed,
            reason=reason,
            output=output,
            eval_time_seconds=elapsed,
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
