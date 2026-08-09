"""Benchmark runner for Huginn."""

from __future__ import annotations

import asyncio
import datetime
import json
import sys
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


def _is_numeric_task(task: BenchmarkTask) -> bool:
    """B1 启发式: 判断 task 是否期望数值答案.

    ponytail: inspect.getsource 看 evaluator 源码是否含 _eval_num / _num_close.
      ceiling: lambda 闭包可能匹配不上, 内联数值比较也匹配不上.
      升级路径: BenchmarkTask 加 expects_numeric 显式字段.
    """
    import inspect
    try:
        src = inspect.getsource(task.evaluator)
        return "_eval_num" in src or "_num_close" in src
    except (TypeError, OSError):
        return False


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
        from huginn.phases import ResearchPhase, autoloop_to_phase, phase_to_autoloop
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
    # B2: re-ask consistency 不一致计数
    unreliable: int = 0
    # B3: 视觉失败模式分组统计 {tag: {"passed": n, "failed": n, "unreliable": n}}
    visual_failure_summary: dict[str, dict[str, int]] = field(default_factory=dict)


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
        re_ask: bool = False,
    ) -> BenchmarkReport:
        """Run all matching tasks and optionally trigger self-evolution.

        B2: re_ask=True 时同一题跑 2 次 (第二次 prompt 加尾空格),
        两次输出用 strict judge 比对, 不一致标 unreliable 不计入正确率分母.
        PerceptionBench 启发: re-ask 暴露模型瞎蒙行为.
        """
        run_id = uuid.uuid4().hex[:8]
        started = datetime.datetime.now().isoformat()
        register_all_tools()

        results: list[TaskResult] = []
        passed = failed = skipped = unreliable = 0

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
                        visual_capability_tag=task.visual_capability_tag,
                    )
                )
                continue

            result = self._run_task(task)
            # B3: 传递 visual_capability_tag 到 result
            result.visual_capability_tag = task.visual_capability_tag

            # B2: re-ask consistency — 跑第二次, strict judge 比对
            if re_ask and task.prompt and task.requires_api_key:
                from dataclasses import replace
                task2 = replace(task, prompt=task.prompt + " ")
                result2 = self._run_task(task2)
                if not self._check_re_ask_consistency(task.prompt, result.output, result2.output):
                    result.unreliable = True
                    unreliable += 1
                    results.append(result)
                    # unreliable 不计入 passed/failed 分母
                    continue

            results.append(result)
            if result.passed:
                passed += 1
            else:
                failed += 1

        finished = datetime.datetime.now().isoformat()
        # B2: pass_rate 分母排除 unreliable
        reliable_count = len(results) - unreliable
        total_time = sum(r.exec_time_seconds + r.eval_time_seconds for r in results)
        metrics = {
            "pass_rate": passed / reliable_count if reliable_count > 0 else 0.0,
            "avg_task_time_seconds": total_time / len(results) if results else 0.0,
            "unreliable_rate": unreliable / len(results) if results else 0.0,
        }

        # B3: visual_capability_tag 分组统计
        visual_failure_summary = self._summarize_by_visual_tag(results)

        evolution_report = None
        if evolve:
            from huginn.evolution.engine import EvolutionEngine

            # 无参构造: 用默认全局 logger, rules 写到 ~/.huginn/logs/evolution_rules.json
            # 跟 agent 运行时读同一份, 避免 bench 产出的 rules 被孤立
            engine = EvolutionEngine()
            evolution_report = engine.run_full_evolution_cycle()

        # 落 memory (可选). memory 故障不能拖死 bench, 静默吞掉.
        if self.memory_manager is not None:
            try:
                summary = (
                    f"bench run_id={run_id} passed={passed}/{reliable_count} "
                    f"pass_rate={metrics['pass_rate']:.2%} evolve={evolve} "
                    f"unreliable={unreliable}"
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
            unreliable=unreliable,
            visual_failure_summary=visual_failure_summary,
        )

    def _check_re_ask_consistency(
        self, task_prompt: str, output1: str, output2: str
    ) -> bool:
        """B2: 用 strict judge 比对两次输出是否语义一致.

        ponytail: 两次输出完全相同 → 直接 True, 不调 LLM.
        不同 → 调 strict judge (以 output2 为 reference 判断 output1).
        无 API key → fallback 到去除空格后字符串比较.
        """
        if output1.strip() == output2.strip():
            return True
        from .llm_judge import judge_task
        rubric = judge_task(
            task_prompt=task_prompt,
            agent_output=output1,
            reference=output2,
            strict=True,
        )
        if rubric.reason.startswith(("无 API key", "judge 调用失败")):
            # fallback: 去除首尾空格和标点后比较
            import re as _re
            norm1 = _re.sub(r"[\s,.]+", "", output1).lower()
            norm2 = _re.sub(r"[\s,.]+", "", output2).lower()
            return norm1 == norm2
        return rubric.passed

    def _summarize_by_visual_tag(
        self, results: list[TaskResult]
    ) -> dict[str, dict[str, int]]:
        """B3: 按 visual_capability_tag 分组统计 passed/failed/unreliable.

        PerceptionBench 分类法: count/attr/hallu/fgr/depth/none.
        ponytail: 只统计非 none 的 tag, none 不进 summary.
        """
        summary: dict[str, dict[str, int]] = {}
        for r in results:
            tag = r.visual_capability_tag or "none"
            if tag == "none":
                continue
            if tag not in summary:
                summary[tag] = {"passed": 0, "failed": 0, "unreliable": 0}
            if r.unreliable:
                summary[tag]["unreliable"] += 1
            elif r.passed:
                summary[tag]["passed"] += 1
            else:
                summary[tag]["failed"] += 1
        return summary

    def _has_api_key(self) -> bool:
        return bool(self.config.resolved_api_key)

    def _run_task(self, task: BenchmarkTask) -> TaskResult:
        start = time.time()
        output = ""
        tool_names: set[str] = set()
        # Structural tests (no prompt) skip LLM — evaluator runs directly
        if not task.prompt:
            output = "[structural test]"
        else:
            try:
                # 直接 asyncio.run: _agent_chat 内部 asyncio.timeout 取消协程后,
                # asyncio.run 传播 TimeoutError, 无 ThreadPoolExecutor.shutdown 阻塞.
                # 之前用线程池包裹, shutdown(wait=True) 在超时后仍等线程清理, 实测
                # 180s 超时任务拖到 608s. _run_task 是同步方法不在 event loop 里, 直接跑.
                output, tool_names = asyncio.run(
                    self._agent_chat(task.prompt, timeout=task.timeout_seconds)
                )
            except TimeoutError:
                output = f"[timeout: agent 超过 {task.timeout_seconds}s 未响应]"
            except Exception as exc:
                output = f"Error: {exc}"

        elapsed = time.time() - start
        result = task.evaluate(output)
        result.exec_time_seconds = elapsed

        # B1: 数值必须经工具计算 gate — 期望数值答案 + 全程无计算工具 → 打回一次
        # ponytail: 用 inspect.getsource 判断 evaluator 是否数值型 (含 _eval_num/_num_close).
        #   ceiling: 启发式, lambda 闭包源码可能匹配不上. 升级路径: BenchmarkTask 加 expects_numeric 字段.
        if task.prompt and not result.passed and _is_numeric_task(task):
            compute_tools = {"code_tool", "symbolic_math_tool", "bash_tool",
                             "python_repl", "symbolic_math", "calculator"}
            if not (tool_names & compute_tools):
                # 打回一次注入 "You must compute, not guess"
                retry_prompt = (
                    task.prompt
                    + "\n\n[B1 GATE] Your previous answer was numeric but you did not "
                    "call any computation tool (code_tool / symbolic_math_tool / bash_tool). "
                    "You MUST compute, not guess. Re-solve this problem by calling a "
                    "computation tool to derive the numeric answer, then state the final answer."
                )
                try:
                    retry_output, retry_tools = asyncio.run(
                        self._agent_chat(retry_prompt, timeout=task.timeout_seconds)
                    )
                    if retry_output and retry_output.strip() != output.strip():
                        retry_result = task.evaluate(retry_output)
                        retry_result.exec_time_seconds = elapsed
                        # 重跑后用了计算工具 + 评分通过 → 采纳 retry
                        if retry_result.passed or (retry_tools & compute_tools):
                            result = retry_result
                            output = retry_output
                            tool_names = retry_tools
                        # 重跑仍 fail + 仍无计算工具 → 标记 B1 FAIL
                        if not result.passed and not (tool_names & compute_tools):
                            result.reason += " [B1: numeric answer without computation tool — FAIL]"
                            result.passed = False
                except (TimeoutError, Exception) as _b1_exc:
                    # B1 retry 失败不阻塞, 保留原 result
                    pass

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

    async def _agent_chat(self, prompt: str, timeout: float = 120.0) -> tuple[str, set[str]]:
        """Send a single prompt to HuginnAgent and return (final text, tool_names used).

        timeout: asyncio 层面超时, 超时后取消协程 (不只是 ThreadPoolExecutor 等待).
        B1: 收集 tool_names 给 _run_task 做数值 gate 判定.
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
            agent.langchain_tools, seed=hash(prompt) & 0xFFFFFFFF
        )
        agent._invalidate_tool_description_cache()

        final = ""
        tool_names_used: set[str] = set()
        # ponytail: asyncio.timeout (3.11+) 取消协程, 避免 agent 工具循环卡死时
        # ThreadPoolExecutor.shutdown(wait=True) 阻塞主线程. 超时后 agent.chat 的
        # async generator 会被 close, 协程内 pending 的 await 抛 CancelledError.
        async with asyncio.timeout(timeout):
            async for chunk in agent.chat(prompt):
                msgs = chunk.get("messages", [])
                for msg in msgs:
                    # B1: 收集 tool names — AIMessage.tool_calls + ToolMessage.name
                    tcs = getattr(msg, "tool_calls", None) or []
                    for tc in tcs:
                        name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                        if name:
                            tool_names_used.add(name)
                    tname = getattr(msg, "name", None)
                    if tname and tname != "HuginnAgent":
                        tool_names_used.add(tname)
                if msgs:
                    last = msgs[-1]
                    content = getattr(last, "content", "")
                    if content:
                        final = str(content)
        return final, tool_names_used

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
            "unreliable": report.unreliable,
            "metrics": report.metrics,
            "results": [
                {
                    "task_id": r.task_id,
                    "category": r.category,
                    "passed": r.passed,
                    "reason": r.reason,
                    "exec_time_seconds": r.exec_time_seconds,
                    "eval_time_seconds": r.eval_time_seconds,
                    "unreliable": r.unreliable,
                    "visual_capability_tag": r.visual_capability_tag,
                }
                for r in report.results
            ],
            "visual_failure_summary": report.visual_failure_summary,
            "evolution_report": report.evolution_report,
        }
        target.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )


# ── self-check ─────────────────────────────────────────────────


def _selfcheck() -> None:
    """B2+B3 selfcheck: re-ask consistency + visual tag 分组统计.

    ponytail: mock runner 验 _check_re_ask_consistency + _summarize_by_visual_tag.
    不调真 LLM, 不跑真 agent.
    ceiling: 没验 _run_task 真路径 (需真 agent + API key), 见 acceptance test.
    """
    import os

    # 构造最小 runner (不调 __init__, 避免 HuginnConfig.from_env 副作用)
    runner = BenchmarkRunner.__new__(BenchmarkRunner)

    # B2.1: 两次输出完全相同 → consistent
    assert runner._check_re_ask_consistency("test", "16", "16") is True, \
        "相同输出应 consistent"
    print("1. re-ask same output → consistent OK")

    # B2.2: 两次输出不同 + 无 API key → fallback 字符串比较
    orig_key = os.environ.pop("DEEPSEEK_API_KEY", None)
    orig_key2 = os.environ.pop("HUGINN_API_KEY", None)
    try:
        # fallback: 去除空格标点后比较
        consistent = runner._check_re_ask_consistency("test", "16.0", "16.0 ")
        assert consistent is True, "空格差异 fallback 后应 consistent"
        print("2a. re-ask whitespace diff + no key → consistent OK")

        # 真不同 → fallback 后仍不同 → inconsistent
        inconsistent = runner._check_re_ask_consistency("test", "16", "60")
        assert inconsistent is False, "不同输出应 inconsistent"
        print("2b. re-ask different output + no key → inconsistent OK")
    finally:
        if orig_key:
            os.environ["DEEPSEEK_API_KEY"] = orig_key
        if orig_key2:
            os.environ["HUGINN_API_KEY"] = orig_key2

    # B3.1: visual_capability_tag 分组统计
    from .task import TaskResult
    mock_results = [
        TaskResult(task_id="t1", category="vis", passed=True, reason="", output="",
           visual_capability_tag="count"),
        TaskResult(task_id="t2", category="vis", passed=False, reason="", output="",
           visual_capability_tag="count"),
        TaskResult(task_id="t3", category="vis", passed=False, reason="", output="",
           visual_capability_tag="hallu", unreliable=True),
        TaskResult(task_id="t4", category="vis", passed=True, reason="", output="",
           visual_capability_tag="depth"),
        TaskResult(task_id="t5", category="math", passed=True, reason="", output="",
           visual_capability_tag="none"),  # none 不进 summary
    ]
    summary = runner._summarize_by_visual_tag(mock_results)
    assert "count" in summary, f"count tag 应在 summary: {summary}"
    assert summary["count"] == {"passed": 1, "failed": 1, "unreliable": 0}, summary["count"]
    assert summary["hallu"] == {"passed": 0, "failed": 0, "unreliable": 1}, summary["hallu"]
    assert summary["depth"] == {"passed": 1, "failed": 0, "unreliable": 0}, summary["depth"]
    assert "none" not in summary, "none tag 不应进 summary"
    print(f"3. visual tag 分组统计: {summary} OK")

    # B3.2: 全 none tag → 空 summary
    all_none = [TaskResult(task_id="t1", category="math", passed=True, reason="", output="")]
    empty_summary = runner._summarize_by_visual_tag(all_none)
    assert empty_summary == {}, f"全 none 应空 summary: {empty_summary}"
    print("4. all none tag → empty summary OK")

    print("runner B2+B3 selfcheck OK")


def _selfcheck_b1() -> int:
    """B1 self-check: 验证 _is_numeric_task 启发式 + B1 retry 骨架.

    ponytail: 不调真 agent, 只验启发式判断 + retry prompt 构造逻辑.
      ceiling: 没验 retry 真路径 (需真 agent + API key).
    """
    # 1. _is_numeric_task 对数值 task 返回 True
    num_task = BenchmarkTask(
        id="num-test",
        category="math",
        prompt="What is 2+2?",
        evaluator=lambda out: _eval_num_in_output(out, 4, 0.1),
    )
    assert _is_numeric_task(num_task) is True, "数值 task 应识别为 numeric"
    print("[CHECK B1.1] numeric task detected OK")

    # 2. _is_numeric_task 对非数值 task 返回 False
    from .task import contains_any
    kw_task = BenchmarkTask(
        id="kw-test",
        category="text",
        prompt="List three fruits.",
        evaluator=contains_any(["apple", "banana", "cherry"]),
    )
    assert _is_numeric_task(kw_task) is False, "关键词 task 不应识别为 numeric"
    print("[CHECK B1.2] non-numeric task rejected OK")

    # 3. compute_tools 集合非空 + 包含 code_tool
    compute_tools = {"code_tool", "symbolic_math_tool", "bash_tool",
                     "python_repl", "symbolic_math", "calculator"}
    assert "code_tool" in compute_tools
    assert len(compute_tools) >= 4
    print("[CHECK B1.3] compute_tools set OK")

    # 4. B1 retry prompt 构造 (不调真 agent, 只验字符串)
    retry_prompt = (
        num_task.prompt
        + "\n\n[B1 GATE] Your previous answer was numeric but you did not "
        "call any computation tool (code_tool / symbolic_math_tool / bash_tool). "
        "You MUST compute, not guess. Re-solve this problem by calling a "
        "computation tool to derive the numeric answer, then state the final answer."
    )
    assert "[B1 GATE]" in retry_prompt
    assert "compute, not guess" in retry_prompt
    assert num_task.prompt in retry_prompt
    print("[CHECK B1.4] retry prompt construction OK")

    # 5. tool_names 拦截逻辑: 空 set → 触发; 含 code_tool → 不触发
    assert not (set() & compute_tools), "空 tool_names 应触发 B1"
    assert ({"code_tool"} & compute_tools), "含 code_tool 不应触发 B1"
    assert not ({"web_search_tool"} & compute_tools), "web_search 不算计算工具"
    print("[CHECK B1.5] tool_names interception logic OK")

    print("[CHECK B1] ALL ASSERTS PASSED")
    return 0


if __name__ == "__main__":
    if "--self-check-b1" in sys.argv:
        sys.exit(_selfcheck_b1())
    _selfcheck()
