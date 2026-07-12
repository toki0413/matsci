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
from dataclasses import asdict, dataclass, field
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
    cost: float = 0.0


@dataclass
class CaseTrialResult:
    """One case run N times — feeds pass^3 / pass@3 computation."""

    case_id: str
    task: str
    category: str
    trials: list[BenchmarkResult]
    pass_all: bool   # pass^3: every trial passed
    pass_any: bool   # pass@3: at least one trial passed
    avg_score: float
    max_score: float


@dataclass
class MultiTrialResult:
    """Suite-level multi-trial result with ClawBench metrics."""

    case_results: list[CaseTrialResult]
    trials: int
    avg_score: float         # S: mean score across all trials, normalized 0-1
    pass_all_rate: float     # pass^3 fraction of cases
    pass_any_rate: float     # pass@3 fraction of cases
    final_score: float       # 100 * S^0.40 * r_all^0.45 * r_any^0.15
    total_cost: float
    avg_latency_ms: float
    coverage: float           # fraction of categories with >=1 pass


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

    def materials_science_research_cases(self) -> BenchmarkSuite:
        """Register 11 cases covering the matsci agent's skill surface.

        Spans structure queries, band gaps, DB retrieval, symbolic math,
        literature review, multiscale modeling, phase diagrams, thermo,
        degradation analysis, research design, and elastic properties.
        """
        # 1. structure query
        self.add(BenchmarkCase(
            task="硅在室温下的晶体结构是什么？给出空间群和晶格常数。",
            expected_keywords=["diamond", "cubic", "fd-3m"],
            category="structure",
            tags=["crystal", "silicon"],
        ))
        # 2. band gap calculation
        self.add(BenchmarkCase(
            task="计算硅的带隙（eV），并说明是直接还是间接带隙。",
            expected_value=1.12,
            tolerance=0.05,
            evaluator=numeric_evaluator,
            category="electronic",
            tags=["bandgap", "silicon"],
        ))
        # 3. materials database retrieval
        self.add(BenchmarkCase(
            task="从Materials Project查询BaTiO3的基本信息：空间群、带隙、形成能。",
            expected_keywords=["batio3", "perovskite", "p4mm"],
            category="database",
            tags=["mp", "query"],
        ))
        # 4. symbolic computation
        self.add(BenchmarkCase(
            task="用SymPy计算 x^2 在区间 [0, 1] 上的定积分。",
            expected_value=0.333,
            tolerance=0.01,
            evaluator=numeric_evaluator,
            category="symbolic",
            tags=["sympy", "integral"],
        ))
        # 5. literature review
        self.add(BenchmarkCase(
            task="综述钙钛矿太阳能电池最近五年的效率进展，列出关键里程碑。",
            expected_keywords=["perovskite", "solar", "efficiency", "25"],
            category="literature",
            tags=["review", "photovoltaic"],
        ))
        # 6. multiscale modeling advice
        self.add(BenchmarkCase(
            task="为Li离子电池正极材料设计多尺度建模方案，从DFT到连续介质。",
            rubric_items=[
                {"criterion": "mentions DFT/ab initio", "weight": 2, "keywords": ["dft", "ab initio"]},
                {"criterion": "mentions molecular dynamics", "weight": 2, "keywords": ["molecular dynamics", "md "]},
                {"criterion": "mentions continuum/FEM", "weight": 2, "keywords": ["continuum", "finite element", "fem"]},
                {"criterion": "mentions scale bridging", "weight": 1, "keywords": ["bridge", "coupling", "handoff", "multiscale"]},
            ],
            evaluator=rubric_evaluator,
            category="multiscale",
            tags=["battery", "modeling"],
        ))
        # 7. phase diagram analysis
        self.add(BenchmarkCase(
            task="分析Fe-C二元相图中的共析反应，给出反应温度和产物。",
            expected_keywords=["eutectoid", "austenite", "pearlite", "727"],
            category="phase_diagram",
            tags=["fe-c", "metallurgy"],
        ))
        # 8. thermodynamic properties
        self.add(BenchmarkCase(
            task="计算水在298 K下的标准生成焓（kJ/mol）。",
            expected_value=-285.8,
            tolerance=2.0,
            evaluator=numeric_evaluator,
            category="thermodynamics",
            tags=["enthalpy", "water"],
        ))
        # 9. degradation mechanism analysis
        self.add(BenchmarkCase(
            task="分析PVDF聚合物在紫外光照射下的降解机制。",
            expected_keywords=["radical", "dehydrofluorination", "chain scission"],
            category="degradation",
            tags=["polymer", "uv"],
        ))
        # 10. research proposal design
        self.add(BenchmarkCase(
            task="设计一个高通量计算筛选固态电解质的流程，包含描述符和筛选标准。",
            rubric_items=[
                {"criterion": "mentions high-throughput workflow", "weight": 2, "keywords": ["high-throughput", "high throughput", "screening"]},
                {"criterion": "mentions ionic conductivity descriptor", "weight": 2, "keywords": ["ionic conductivity", "conductivity"]},
                {"criterion": "mentions stability criterion", "weight": 2, "keywords": ["stability", "electrochemical window"]},
                {"criterion": "mentions candidate material families", "weight": 1, "keywords": ["li7p3s11", "lgps", "argyrodite", "garnet", "llzo"]},
            ],
            evaluator=rubric_evaluator,
            category="research_design",
            tags=["high-throughput", "electrolyte"],
        ))
        # 11. elastic properties (bonus)
        self.add(BenchmarkCase(
            task="计算铜的体弹模量（GPa）。",
            expected_value=140.0,
            tolerance=15.0,
            evaluator=numeric_evaluator,
            category="mechanical",
            tags=["elastic", "copper"],
        ))
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

    async def run_multi_trial(
        self,
        agent: Any,
        trials: int = 3,
        thread_id: str = "benchmark",
        checkpoint_path: str | None = None,
    ) -> MultiTrialResult:
        """Run each case ``trials`` times and compute pass^3 / pass@3 / FinalScore.

        If *checkpoint_path* is set, completed trials are persisted after each
        round so a crash or Ctrl-C can resume without redoing finished work.
        """
        from pathlib import Path

        trial_runs: list[list[BenchmarkResult]] = []

        # --- resume from checkpoint ---
        if checkpoint_path and Path(checkpoint_path).exists():
            saved = json.loads(Path(checkpoint_path).read_text(encoding="utf-8"))
            for run_data in saved.get("trial_runs", []):
                trial_runs.append([BenchmarkResult(**r) for r in run_data])

        start_trial = len(trial_runs)
        for t in range(start_trial, trials):
            run_results = await self.run(agent, thread_id=f"{thread_id}_t{t}")
            trial_runs.append(run_results)
            if checkpoint_path:
                Path(checkpoint_path).write_text(
                    json.dumps(
                        {"trial_runs": [[asdict(r) for r in run] for run in trial_runs]},
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )

        return self._compile_multi_trial(trial_runs[:trials], trials)

    def _compile_multi_trial(
        self,
        trial_runs: list[list[BenchmarkResult]],
        trials: int,
    ) -> MultiTrialResult:
        """Group trial runs by case and compute ClawBench metrics."""
        all_scores: list[float] = []
        all_latencies: list[float] = []
        total_cost = 0.0
        covered_cats: set[str] = set()

        case_results: list[CaseTrialResult] = []
        for i, case in enumerate(self.cases):
            trial_results = [run[i] for run in trial_runs]
            successes = [r.success for r in trial_results]
            scores = [r.score for r in trial_results]

            # normalize rubric scores (0-100) to 0-1 for the FinalScore formula
            norm = [s / 100.0 if s > 1.0 else s for s in scores]
            all_scores.extend(norm)
            all_latencies.extend(r.duration_ms for r in trial_results)
            total_cost += sum(getattr(r, "cost", 0.0) for r in trial_results)

            pass_all = all(successes)
            pass_any = any(successes)
            if pass_any:
                covered_cats.add(case.category)

            case_results.append(CaseTrialResult(
                case_id=case.case_id,
                task=case.task,
                category=case.category,
                trials=trial_results,
                pass_all=pass_all,
                pass_any=pass_any,
                avg_score=round(sum(scores) / len(scores), 3) if scores else 0.0,
                max_score=max(scores) if scores else 0.0,
            ))

        n = len(self.cases) or 1
        pass_all_rate = sum(1 for cr in case_results if cr.pass_all) / n
        pass_any_rate = sum(1 for cr in case_results if cr.pass_any) / n

        S = sum(all_scores) / len(all_scores) if all_scores else 0.0
        r_all = pass_all_rate ** (1.0 / 3.0)
        r_any = 1.0 - (1.0 - pass_any_rate) ** (1.0 / 3.0)
        final_score = round(100.0 * (S ** 0.40) * (r_all ** 0.45) * (r_any ** 0.15), 2)

        all_cats = {c.category for c in self.cases}
        coverage = len(covered_cats) / len(all_cats) if all_cats else 0.0

        return MultiTrialResult(
            case_results=case_results,
            trials=trials,
            avg_score=round(S, 4),
            pass_all_rate=round(pass_all_rate, 4),
            pass_any_rate=round(pass_any_rate, 4),
            final_score=final_score,
            total_cost=round(total_cost, 4),
            avg_latency_ms=round(sum(all_latencies) / len(all_latencies), 2) if all_latencies else 0.0,
            coverage=round(coverage, 4),
        )

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

    def summary(
        self,
        results: list[BenchmarkResult] | MultiTrialResult,
    ) -> dict[str, Any]:
        """Return aggregate statistics for a result set.

        Pass a :class:`MultiTrialResult` to get ClawBench FinalScore and
        multi-trial metrics (pass^3, pass@3, coverage, cost).
        """
        if isinstance(results, MultiTrialResult):
            return {
                "trials": results.trials,
                "total_cases": len(results.case_results),
                "avg_score": results.avg_score,
                "pass_all_rate": results.pass_all_rate,   # pass^3
                "pass_any_rate": results.pass_any_rate,   # pass@3
                "final_score": results.final_score,
                "total_cost": results.total_cost,
                "avg_latency_ms": results.avg_latency_ms,
                "coverage": results.coverage,
                "case_results": [
                    {
                        "case_id": cr.case_id,
                        "task": cr.task,
                        "category": cr.category,
                        "pass_all": cr.pass_all,
                        "pass_any": cr.pass_any,
                        "avg_score": cr.avg_score,
                        "max_score": cr.max_score,
                        "n_trials": len(cr.trials),
                    }
                    for cr in results.case_results
                ],
            }

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
