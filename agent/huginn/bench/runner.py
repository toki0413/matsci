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


def _eval_num_in_output(output: str, expected: float, tol: float) -> tuple[bool, str]:
    """从 agent 输出里提取数值, 容差匹配. 不再依赖前 N 字符截断."""
    import re
    # 匹配带小数/科学计数法的数值
    nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", output)]
    if not nums:
        return False, f"未找到数值 (期望 {expected})"
    for n in nums:
        if abs(n - expected) <= tol:
            return True, f"got {n} (期望 {expected}±{tol})"
    return False, f"got {nums[0]}, 期望 {expected}±{tol}"


DEFAULT_TASKS: list[BenchmarkTask] = [
    BenchmarkTask(
        id="math-simple",
        category="math",
        prompt="What is the value of (3 + 5) * 2? Reply with only the number.",
        evaluator=lambda out: _eval_num_in_output(out, 16, 0.5),
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
        evaluator=lambda out: _eval_num_in_output(out, 60, 0.5),
        tags=["materials", "elasticity"],
        requires_api_key=True,
    ),
    BenchmarkTask(
        id="code-function",
        category="coding",
        prompt="Write a Python function `bulk_modulus(c11, c12)` that returns (c11 + 2*c12) / 3. Reply with only the code block.",
        evaluator=lambda out: (
            "def bulk_modulus" in out and "c11 + 2*c12" in out.replace(" ", ""),
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
            and (
                "x ^ 2 + 3 * x" in out.replace("**", "^").replace(" ", "")
                or "x**2 + 3*x" in out.replace(" ", "")
            ),
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
    # ── GPQA/HLE 式知识推理题 (对标 MMMU/GPQA, 需要 API key) ────
    BenchmarkTask(
        id="knowledge-silicon-bandgap",
        category="knowledge",
        prompt="硅的室温带隙是多少 eV? 只回答数值.",
        evaluator=lambda out: _eval_num_in_output(out, 1.12, 0.1),
        tags=["knowledge", "semiconductor"],
        requires_api_key=True,
        reference="Si 室温带隙 = 1.12 eV (间接带隙)",
    ),
    BenchmarkTask(
        id="knowledge-copper-conductivity",
        category="knowledge",
        prompt="铜在 20°C 的电导率是多少 MS/m? 只回答数值.",
        evaluator=lambda out: _eval_num_in_output(out, 59.6, 2.0),
        tags=["knowledge", "metals"],
        requires_api_key=True,
        reference="Cu 电导率 ≈ 59.6 MS/m (IACS 标准 100%)",
    ),
    BenchmarkTask(
        id="knowledge-iron-bcc",
        category="knowledge",
        prompt="室温下铁的晶体结构是什么? 回答英文缩写 (BCC/FCC/HCP).",
        evaluator=lambda out: ("BCC" in out.upper()[:20], "应为 BCC"),
        tags=["knowledge", "crystal"],
        requires_api_key=True,
        reference="室温铁是 BCC (α-Fe), 912°C 以上转 FCC (γ-Fe)",
    ),
    BenchmarkTask(
        id="knowledge-avogadro",
        category="knowledge",
        prompt="阿伏伽德罗常数是多少 (×10²³ mol⁻¹)? 只回答数值.",
        evaluator=lambda out: _eval_num_in_output(out, 6.022, 0.01),
        tags=["knowledge", "chemistry"],
        requires_api_key=True,
        reference="NA = 6.022×10²³ mol⁻¹",
    ),
    BenchmarkTask(
        id="knowledge-boltzmann",
        category="knowledge",
        prompt="玻尔兹曼常数是多少 (×10⁻²³ J/K)? 只回答数值.",
        evaluator=lambda out: _eval_num_in_output(out, 1.381, 0.01),
        tags=["knowledge", "physics"],
        requires_api_key=True,
        reference="kB = 1.381×10⁻²³ J/K",
    ),
    BenchmarkTask(
        id="knowledge-graphite-density",
        category="knowledge",
        prompt="石墨的密度是多少 g/cm³? 只回答数值.",
        evaluator=lambda out: _eval_num_in_output(out, 2.27, 0.1),
        tags=["knowledge", "carbon"],
        requires_api_key=True,
        reference="石墨密度 ≈ 2.27 g/cm³",
    ),
    BenchmarkTask(
        id="knowledge-water-boiling",
        category="knowledge",
        prompt="标准大气压下水的沸点是多少 °C? 只回答数值.",
        evaluator=lambda out: _eval_num_in_output(out, 100, 1),
        tags=["knowledge", "physics"],
        requires_api_key=True,
        reference="100°C (1 atm)",
    ),
    BenchmarkTask(
        id="knowledge-nacl-structure",
        category="knowledge",
        prompt="NaCl 晶体的晶格结构是什么? 回答英文 (Rock Salt/Fluorite/Zinc Blende/Diamond).",
        evaluator=lambda out: ("rock salt" in out.lower() or "rocksalt" in out.lower() or "nacl" in out.lower()[:10], "应为 Rock Salt"),
        tags=["knowledge", "crystal"],
        requires_api_key=True,
        reference="NaCl 是 Rock Salt 结构 (FCC, 空间群 Fm-3m)",
    ),
    BenchmarkTask(
        id="knowledge-planck",
        category="knowledge",
        prompt="普朗克常数 h 是多少 (×10⁻³⁴ J·s)? 只回答数值.",
        evaluator=lambda out: _eval_num_in_output(out, 6.626, 0.01),
        tags=["knowledge", "physics"],
        requires_api_key=True,
        reference="h = 6.626×10⁻³⁴ J·s",
    ),
    BenchmarkTask(
        id="knowledge-diamond-bandgap",
        category="knowledge",
        prompt="金刚石(钻石)的带隙是多少 eV? 只回答数值.",
        evaluator=lambda out: _eval_num_in_output(out, 5.5, 0.3),
        tags=["knowledge", "carbon"],
        requires_api_key=True,
        reference="金刚石带隙 ≈ 5.5 eV (间接带隙, 绝缘体)",
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
        memory_manager: Any = None,
    ):
        self.tasks = tasks or DEFAULT_TASKS
        self.config = config or HuginnConfig.from_env()
        self.logger = logger or ExecutionLogger()
        # ponytail: memory_manager 可选, 不破坏现有调用; 升级路径是 bench 自动注入 agent 的 memory
        self.memory_manager = memory_manager

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

        # 落 memory (可选). memory 故障不能拖死 bench, 静默吞掉.
        if self.memory_manager is not None:
            try:
                summary = (
                    f"bench run_id={run_id} passed={passed}/{len(results)} "
                    f"pass_rate={metrics['pass_rate']:.2%} evolve={evolve}"
                )
                if hasattr(self.memory_manager, "remember"):
                    self.memory_manager.remember(
                        content=summary,
                        category="benchmark_run_summary",
                        importance=0.8,
                    )
                elif hasattr(self.memory_manager, "store"):
                    self.memory_manager.store(
                        content=summary,
                        category="benchmark_run_summary",
                        importance=0.8,
                    )
            except Exception:
                pass

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
                # 直接 asyncio.run: _agent_chat 内部 asyncio.timeout 取消协程后,
                # asyncio.run 传播 TimeoutError, 无 ThreadPoolExecutor.shutdown 阻塞.
                # 之前用线程池包裹, shutdown(wait=True) 在超时后仍等线程清理, 实测
                # 180s 超时任务拖到 608s. _run_task 是同步方法不在 event loop 里, 直接跑.
                output = asyncio.run(
                    self._agent_chat(task.prompt, timeout=task.timeout_seconds)
                )
            except TimeoutError:
                output = f"[timeout: agent 超过 {task.timeout_seconds}s 未响应]"
            except Exception as exc:
                output = f"Error: {exc}"

        elapsed = time.time() - start
        result = task.evaluate(output)
        result.exec_time_seconds = elapsed

        # LLM judge: regex 评分低时触发二次评审 (对标 PaperBench SimpleJudge)
        if task.prompt and task.reference is not None:
            from .llm_judge import judge_with_regex_fallback
            result = judge_with_regex_fallback(
                task_prompt=task.prompt,
                agent_output=output,
                regex_result=result,
                reference=task.reference,
                is_code_task=task.is_code_task,
            )

        self.logger.log_conversation(
            session_id=f"bench-{task.id}",
            user_message=task.prompt,
            agent_response=output,
            topic_tags=task.tags,
        )
        return result

    async def _agent_chat(self, prompt: str, timeout: float = 120.0) -> str:
        """Send a single prompt to HuginnAgent and return the final assistant text.

        timeout: asyncio 层面超时, 超时后取消协程 (不只是 ThreadPoolExecutor 等待).
        """
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

        from huginn.prompts import MATH_DEPTH_GUIDE

        # 第二轮 80% 的 prompt (已验证最优), 只加一句验证约束.
        # concise 是数学之美: 答案简洁, 但推导必须完整.
        bench_system_prompt = (
            "You are a scientific research assistant solving challenging "
            "physics/chemistry/materials problems. Take your time and reason "
            "thoroughly.\n\n"
            "## Problem-Solving Strategy\n"
            "1. Decompose the question — identify what's given and what's asked\n"
            "2. Recall relevant theory (formula, principle, definition)\n"
            "3. Use code_tool to compute when arithmetic is non-trivial\n"
            "4. Use web_search to verify constants, definitions, or edge cases\n"
            "5. Cross-check the answer via an independent method if possible\n"
            "6. Only then commit to a final answer\n\n"
            "## Rules\n"
            "- Use tools aggressively — do not guess when you can compute or search\n"
            "- Show intermediate steps so reasoning is auditable\n"
            "- For multiple choice: eliminate wrong options first, then verify\n"
            "- Even when the answer seems obvious, verify before committing. "
            "Conciseness is for the final answer, not the derivation\n"
        ) + MATH_DEPTH_GUIDE

        agent = HuginnAgent(
            model=model,
            system_prompt=bench_system_prompt,
            memory_manager=None,
            max_tool_output_tokens=self.config.max_tool_output_tokens,
            context_budget_tokens=self.config.context_budget_tokens,
            # 深度模式: 拉长工具链但防 overflow. 30 会导致 context 爆炸.
            max_tool_calls=20,
            max_tool_calls_per_tool=8,
        )
        agent.register_tools_from_registry()

        # Inkling 启发: 打乱工具顺序, 防 agent 对工具位置过拟合.
        # 每 task 用不同 seed, 保证顺序不同但可复现.
        # ponytail: 只在 bench 层做, 生产 run() 不受影响.
        from .tool_randomization import randomize_tool_order

        agent.langchain_tools = randomize_tool_order(
            agent.langchain_tools, seed=hash(task.id) & 0xFFFFFFFF
        )
        agent._invalidate_tool_description_cache()

        final = ""
        # ponytail: asyncio.timeout (3.11+) 取消协程, 避免 agent 工具循环卡死时
        # ThreadPoolExecutor.shutdown(wait=True) 阻塞主线程. 超时后 agent.chat 的
        # async generator 会被 close, 协程内 pending 的 await 抛 CancelledError.
        async with asyncio.timeout(timeout):
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
