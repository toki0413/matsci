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

__all__ = ["BenchmarkRunner", "BenchmarkReport", "BenchmarkTask", "TaskResult", "get_suite_tasks"]


def get_suite_tasks(suite: str) -> list[BenchmarkTask]:
    """按 suite 名返回 BenchmarkTask 列表. research 返回空 (走独立脚本)."""
    if suite == "general":
        from .runner import DEFAULT_TASKS
        return list(DEFAULT_TASKS)
    if suite == "lineage":
        from .ideagene_bench import build_ideagene_tasks
        return build_ideagene_tasks()
    if suite == "repro":
        from .paper_repro_bench import build_repro_tasks
        return build_repro_tasks()
    if suite == "optim":
        from .optim_bench import build_optim_tasks
        return build_optim_tasks()
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

