"""Benchmark harness for Huginn.

6 个能力层对标社区 AI Scientist 评测:
  - general:  知识推理+工具 (MMMU/GPQA 式)
  - physics:  数值物理正确性 (MatWorldBench)
  - research: 研究场景 (RCBench, 多 trial, 走独立脚本)
  - lineage:  谱系能力 (IdeaGene-Bench, 对标 IG-Bench)
  - repro:    论文复现 (PaperReproBench, 对标 PaperBench)
  - optim:    算法优化 (OptimBench, 对标 MLE-Bench)
"""

from .runner import BenchmarkReport, BenchmarkRunner
from .task import BenchmarkTask, TaskResult
from .llm_judge import judge_task, judge_with_regex_fallback, JudgeRubric
from .baselines import BASELINES, get_baselines_for_suite, format_baseline_table

__all__ = [
    "BenchmarkRunner", "BenchmarkReport", "BenchmarkTask", "TaskResult",
    "get_suite_tasks", "judge_task", "judge_with_regex_fallback", "JudgeRubric",
    "BASELINES", "get_baselines_for_suite", "format_baseline_table",
]


def get_suite_tasks(suite: str, max_tasks: int | None = None) -> list[BenchmarkTask]:
    """按 suite 名返回 BenchmarkTask 列表.

    suite:
      - general:    原有结构测试题 + 知识题 (20)
      - mmlu:       MMLU 科学学科 (默认 500, 对标 MMMU/GPQA)
      - sciq:       SciQ 自然科学 (默认 500)
      - arc:        ARC-Challenge 科学推理 (默认 500)
      - gpqa:       GPQA PhD 级 (ModelScope, 默认 100, HF 上 gated)
      - cmmlu:      CMMLU 中文科学 (ModelScope, 默认 500)
      - mmlu_pro:   MMLU-Pro 10 选项更难 (ModelScope, 默认 500)
      - external:   全部外部数据集合并 (MMLU+SciQ+ARC+GPQA+CMMLU+MMLU-Pro)
      - physics:    MatWorldBench adapter (10)
      - lineage:    IdeaGene-Bench (15)
      - repro:      PaperReproBench (10)
      - optim:      OptimBench (8)
      - naturebench: NatureBench-mini (10, Nature 系列论文复现)
      - research:   走独立脚本, 返回空

    max_tasks: 限制题量 (从全量随机抽样), None 表示不限制.
    """
    if suite == "general":
        from .runner import DEFAULT_TASKS
        return list(DEFAULT_TASKS)
    if suite == "mmlu":
        from .adapters import load_mmlu_tasks
        return load_mmlu_tasks(max_tasks=max_tasks or 500)
    if suite == "sciq":
        from .adapters import load_sciq_tasks
        return load_sciq_tasks(max_tasks=max_tasks or 500)
    if suite == "arc":
        from .adapters import load_arc_tasks
        return load_arc_tasks(max_tasks=max_tasks or 500)
    if suite == "gpqa":
        from .adapters import load_gpqa_tasks
        return load_gpqa_tasks(max_tasks=max_tasks or 100)
    if suite == "cmmlu":
        from .adapters import load_cmmlu_tasks
        return load_cmmlu_tasks(max_tasks=max_tasks or 500)
    if suite == "mmlu_pro":
        from .adapters import load_mmlu_pro_tasks
        return load_mmlu_pro_tasks(max_tasks=max_tasks or 500)
    if suite == "external":
        from .adapters import load_all_external
        return load_all_external(max_per_dataset=max_tasks or 400)
    if suite == "lineage":
        from .ideagene_bench import build_ideagene_tasks
        return build_ideagene_tasks()
    if suite == "repro":
        from .paper_repro_bench import build_repro_tasks
        return build_repro_tasks()
    if suite == "optim":
        from .optim_bench import build_optim_tasks
        return build_optim_tasks()
    if suite == "naturebench":
        from .naturebench import build_naturebench_tasks
        return build_naturebench_tasks()
    if suite == "physics":
        return _build_physics_tasks()
    return []


def _build_physics_tasks() -> list[BenchmarkTask]:
    """MatWorldBench adapter: 转成 BenchmarkTask 列表."""
    import re
    from huginn.evaluation.matworld_bench import MatWorldBench

    tasks: list[BenchmarkTask] = []
    for bt in MatWorldBench.TASKS:
        key = next(iter(bt.expected_result))
        expected = bt.expected_result[key]
        tol = bt.tolerance.get(key, 0.0)

        def _make_eval(exp: float, tolerance: float, k: str):
            def evaluate(output: str) -> tuple[bool, str, float]:
                nums = [float(x) for x in re.findall(
                    r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", output
                )]
                if not nums:
                    return False, f"未找到数值 ({k})", 0.0
                closest = min(nums, key=lambda n: abs(n - exp))
                if abs(closest - exp) <= tolerance:
                    return True, f"{k}={closest} (期望 {exp}±{tolerance})", 1.0
                return False, f"{k}={closest}, 期望 {exp}±{tolerance}", 0.3
            return evaluate

        tasks.append(BenchmarkTask(
            id=bt.id,
            category=bt.category,
            prompt=bt.prompt,
            evaluator=_make_eval(expected, tol, key),
            tags=["physics"] + list(bt.metadata.values()),
            requires_api_key=True,
        ))
    return tasks

