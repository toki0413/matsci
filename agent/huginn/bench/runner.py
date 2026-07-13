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
        evaluator=lambda out: ("16" in out.strip()[:10], f"got {out!r}"),
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
        evaluator=lambda out: ("60" in out.strip()[:10], f"got {out!r}"),
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
    # ── Structural tests (no API key needed) ────────────────────
    BenchmarkTask(
        id="gov-block-dangerous",
        category="governance",
        prompt="",
        evaluator=lambda out: _eval_gov_block(),
        tags=["governance", "security"],
        requires_api_key=False,
    ),
    BenchmarkTask(
        id="clarify-no-false-positive",
        category="clarification",
        prompt="",
        evaluator=lambda out: _eval_clarify_regex(),
        tags=["clarification", "regex"],
        requires_api_key=False,
    ),
    BenchmarkTask(
        id="phase-adapter",
        category="architecture",
        prompt="",
        evaluator=lambda out: _eval_phase_adapter(),
        tags=["phases", "adapter"],
        requires_api_key=False,
    ),
    BenchmarkTask(
        id="ontology-predictability",
        category="governance",
        prompt="",
        evaluator=lambda out: _eval_ontology_pred(),
        tags=["ontology", "predictability"],
        requires_api_key=False,
    ),
    BenchmarkTask(
        id="task-state-tracker",
        category="architecture",
        prompt="",
        evaluator=lambda out: _eval_task_state(),
        tags=["task_state", "long-chain"],
        requires_api_key=False,
    ),
    BenchmarkTask(
        id="kg-feedback-bridge",
        category="validation",
        prompt="",
        evaluator=lambda out: _eval_kg_feedback(),
        tags=["validation", "knowledge_graph"],
        requires_api_key=False,
    ),
]


def _eval_gov_block() -> tuple[bool, str]:
    """Governance blocks dangerous actions without structure context."""
    try:
        from huginn.ontology.actions import get_action_type
        at = get_action_type("run_dft")
        if not at:
            return False, "run_dft action type not registered"
        # No structure provided — should be blocked
        allowed, reasons = at.can_execute({})
        if allowed:
            return False, "run_dft allowed without structure — preconditions not working"
        return True, f"correctly blocked: {reasons[0]}"
    except Exception as e:
        return False, f"governance eval error: {e}"


def _eval_clarify_regex() -> tuple[bool, str]:
    """Clarification regex doesn't false-positive on 'direct or indirect'."""
    import re
    pattern = re.compile(
        r"\beither\s+\w+\s+or\b|\bwhich\b.*\bbetter\b|\bvs\.?\b|\boption\s+[A-C]\b",
        re.IGNORECASE,
    )
    should_not_match = [
        "Calculate the band gap of silicon",
        "What is the direct or indirect band gap of GaAs?",
        "Should I use DFT or MD for this problem?",
    ]
    should_match = [
        "Which is better: VASP or Quantum ESPRESSO?",
        "Compare DFT vs MD approaches",
    ]
    for text in should_not_match:
        if pattern.search(text):
            return False, f"false positive on: {text!r}"
    for text in should_match:
        if not pattern.search(text):
            return False, f"should have matched: {text!r}"
    return True, "5/5 regex checks passed"


def _eval_phase_adapter() -> tuple[bool, str]:
    """Phase adapter maps autoloop ↔ ResearchPhase correctly."""
    try:
        from huginn.phases import autoloop_to_phase, phase_to_autoloop, ResearchPhase
        assert autoloop_to_phase("perceive") == ResearchPhase.LITERATURE
        assert autoloop_to_phase("hypothesize") == ResearchPhase.HYPOTHESIS
        assert autoloop_to_phase("plan") == ResearchPhase.PLANNING
        assert autoloop_to_phase("execute") == ResearchPhase.EXECUTION
        assert autoloop_to_phase("validate") == ResearchPhase.VALIDATION
        assert autoloop_to_phase("report") == ResearchPhase.REPORTING
        assert phase_to_autoloop(ResearchPhase.LITERATURE) == "perceive"
        assert phase_to_autoloop(ResearchPhase.REPORTING) == "report"
        return True, "all 8 adapter mappings correct"
    except Exception as e:
        return False, f"adapter error: {e}"


def _eval_ontology_pred() -> tuple[bool, str]:
    """Action predictability score reflects constraint violations."""
    try:
        from huginn.ontology.actions import get_action_type
        at = get_action_type("run_dft")
        if not at:
            return False, "run_dft not found"
        # With good context — high predictability
        good_ctx = {"energy": -10.5, "max_force": 0.005, "structure": "Si", "encut": 400}
        p_good = at.predictability(good_ctx)
        # With bad context — low predictability (energy positive, force huge)
        bad_ctx = {"energy": 5.0, "max_force": 2.0, "structure": "Si", "encut": 400}
        p_bad = at.predictability(bad_ctx)
        if p_good <= p_bad:
            return False, f"predictability not lower for bad ctx: {p_good:.2f} vs {p_bad:.2f}"
        return True, f"predictability: good={p_good:.2f}, bad={p_bad:.2f}"
    except Exception as e:
        return False, f"ontology error: {e}"


def _eval_task_state() -> tuple[bool, str]:
    """TaskStateTracker records steps and generates context block."""
    try:
        from huginn.memory.task_state import get_tracker
        tracker = get_tracker()
        test_tid = "bench-test-thread"
        # clean up any leftover state from previous runs
        import os
        f = tracker.state_dir / f"{test_tid}.json"
        if f.exists():
            os.remove(f)
        tracker._cache.pop(test_tid, None)

        tracker.record_step(test_tid, action="test action", tool="test_tool",
                            result="test result", findings="test finding")
        state = tracker.get(test_tid)
        if not state.steps:
            return False, "no steps recorded"
        if len(state.steps) != 1:
            return False, f"expected 1 step, got {len(state.steps)}"
        ctx = tracker.context_block(test_tid)
        if "test action" not in ctx and "test_tool" not in ctx:
            return False, "context block missing step info"
        # cleanup
        if f.exists():
            os.remove(f)
        tracker._cache.pop(test_tid, None)
        return True, "step recorded + context block generated"
    except Exception as e:
        return False, f"task_state error: {e}"


def _eval_kg_feedback() -> tuple[bool, str]:
    """KG feedback bridge module imports and function exists."""
    try:
        from huginn.validation.kg_feedback import write_validation_to_kg
        # Just verify it's callable — full test requires a running KG
        result = write_validation_to_kg([], material="Si")
        if result != 0:
            return False, f"expected 0 entries with empty input, got {result}"
        return True, "kg_feedback module functional"
    except Exception as e:
        return False, f"kg_feedback error: {e}"


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
        # Structural tests (no prompt) skip LLM — evaluator runs directly
        if not task.prompt:
            output = "[structural test]"
        else:
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
