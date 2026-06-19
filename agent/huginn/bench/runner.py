"""Benchmark runner for Huginn."""

from __future__ import annotations

import asyncio
import datetime
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from huginn.agent import HuginnAgent
from huginn.config import HuginnConfig
from huginn.evolution.logger import ExecutionLogger
from huginn.models.registry import ModelRegistry
from huginn.tools import register_all_tools

from .task import BenchmarkTask, TaskResult

DEFAULT_TASKS: list[BenchmarkTask] = [
    BenchmarkTask(
        id="math-simple",
        category="math",
        prompt="What is the value of (3 + 5) * 2? Reply with only the number.",
        evaluator=lambda out: (out.strip() == "16", f"got {out!r}"),
        tags=["math", "easy"],
        requires_api_key=True,
    ),
    BenchmarkTask(
        id="materials-bulk-modulus",
        category="materials-science",
        prompt=(
            "The elastic constants of a cubic crystal are c11=100 GPa, c12=40 GPa. "
            "What is the bulk modulus in GPa? Reply with only the number."
        ),
        evaluator=lambda out: ("60" in out.strip() or "60.0" in out, f"got {out!r}"),
        tags=["materials", "elasticity"],
        requires_api_key=True,
    ),
    BenchmarkTask(
        id="code-function",
        category="coding",
        prompt="Write a Python function `bulk_modulus(c11, c12)` that returns (c11 + 2*c12) / 3. Reply with only the code block.",
        evaluator=lambda out: (
            "def bulk_modulus" in out and "(c11 + 2*c12) / 3" in out,
            "missing function or formula",
        ),
        tags=["coding", "python"],
        requires_api_key=True,
    ),
    BenchmarkTask(
        id="symbolic-to-lean",
        category="formal",
        prompt=(
            "Translate the expression 'x**2 + 3*x' into a Lean 4 Float definition named f. "
            "Reply with only the Lean code block."
        ),
        evaluator=lambda out: (
            "def f" in out
            and "Float" in out
            and "x ^ 2 + 3 * x" in out.replace("**", "^"),
            "missing Lean definition or incorrect body",
        ),
        tags=["lean", "formal"],
        requires_api_key=True,
    ),
]


@dataclass
class BenchmarkReport:
    """Report summarizing a benchmark run."""

    run_id: str
    started_at: str
    finished_at: str
    total: int
    passed: int
    failed: int
    skipped: int
    results: list[TaskResult]
    metrics: dict[str, float] = field(default_factory=dict)
    evolution_report: dict[str, Any] | None = None


class BenchmarkRunner:
    """Run a suite of benchmark tasks against Huginn."""

    def __init__(
        self,
        tasks: list[BenchmarkTask] | None = None,
        config: HuginnConfig | None = None,
        logger: ExecutionLogger | None = None,
    ):
        self.tasks = tasks or DEFAULT_TASKS
        self.config = config or HuginnConfig.from_env()
        self.logger = logger or ExecutionLogger()

    def run(
        self,
        evolve: bool = False,
        categories: list[str] | None = None,
    ) -> BenchmarkReport:
        """Run all matching tasks and optionally trigger self-evolution."""
        run_id = uuid.uuid4().hex[:8]
        started = datetime.datetime.now().isoformat()
        register_all_tools()

        results: list[TaskResult] = []
        passed = failed = skipped = 0

        for task in self.tasks:
            if categories and task.category not in categories:
                continue
            if task.requires_api_key and not self._has_api_key():
                skipped += 1
                results.append(
                    TaskResult(
                        task_id=task.id,
                        category=task.category,
                        passed=False,
                        reason="skipped: no API key configured",
                        output="",
                    )
                )
                continue

            result = self._run_task(task)
            results.append(result)
            if result.passed:
                passed += 1
            else:
                failed += 1

        finished = datetime.datetime.now().isoformat()
        total_time = sum(r.exec_time_seconds + r.eval_time_seconds for r in results)
        metrics = {
            "pass_rate": passed / len(results) if results else 0.0,
            "avg_task_time_seconds": total_time / len(results) if results else 0.0,
        }

        evolution_report = None
        if evolve:
            from huginn.evolution.engine import EvolutionEngine

            engine = EvolutionEngine(logger=self.logger)
            evolution_report = engine.run_full_evolution_cycle()

        return BenchmarkReport(
            run_id=run_id,
            started_at=started,
            finished_at=finished,
            total=len(results),
            passed=passed,
            failed=failed,
            skipped=skipped,
            results=results,
            metrics=metrics,
            evolution_report=evolution_report,
        )

    def _has_api_key(self) -> bool:
        return bool(self.config.resolved_api_key)

    def _run_task(self, task: BenchmarkTask) -> TaskResult:
        start = time.time()
        output = ""
        try:
            output = asyncio.run(self._agent_chat(task.prompt))
        except Exception as exc:
            output = f"Error: {exc}"

        elapsed = time.time() - start
        result = task.evaluate(output)
        result.exec_time_seconds = elapsed

        self.logger.log_conversation(
            session_id=f"bench-{task.id}",
            user_message=task.prompt,
            agent_response=output,
            topic_tags=task.tags,
        )
        return result

    async def _agent_chat(self, prompt: str) -> str:
        """Send a single prompt to HuginnAgent and return the final assistant text."""
        registry = ModelRegistry.from_config(self.config)
        alias = registry.default_alias()
        if alias:
            model = registry.resolve(alias)
        elif self.config.provider and self.config.provider != "default":
            model = registry.resolve(
                f"{self.config.provider}/{self.config.model or 'auto'}"
            )
        else:
            raise RuntimeError(
                "No model configured. Set HUGINN_PROVIDER and HUGINN_API_KEY."
            )

        agent = HuginnAgent(
            model=model,
            system_prompt="You are a concise research assistant. Follow instructions exactly.",
            memory_manager=None,
            max_tool_output_tokens=self.config.max_tool_output_tokens,
            context_budget_tokens=self.config.context_budget_tokens,
        )
        agent.register_tools_from_registry()

        final = ""
        async for chunk in agent.chat(prompt):
            msgs = chunk.get("messages", [])
            if msgs:
                last = msgs[-1]
                content = getattr(last, "content", "")
                if content:
                    final = str(content)
        return final

    def save_report(self, report: BenchmarkReport, path: str | Path) -> None:
        """Save a benchmark report to a JSON file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "run_id": report.run_id,
            "started_at": report.started_at,
            "finished_at": report.finished_at,
            "total": report.total,
            "passed": report.passed,
            "failed": report.failed,
            "skipped": report.skipped,
            "metrics": report.metrics,
            "results": [
                {
                    "task_id": r.task_id,
                    "category": r.category,
                    "passed": r.passed,
                    "reason": r.reason,
                    "exec_time_seconds": r.exec_time_seconds,
                    "eval_time_seconds": r.eval_time_seconds,
                }
                for r in report.results
            ],
            "evolution_report": report.evolution_report,
        }
        target.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
