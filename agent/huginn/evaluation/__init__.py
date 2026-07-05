"""Evaluation — MCDA 决策分析 + LLM-as-judge arena + 统一 Grader."""

from huginn.evaluation.arena_store import ArenaRecord, ArenaStore
from huginn.evaluation.core import evaluate
from huginn.evaluation.grader import (
    BenchGrader,
    DimensionalGrader,
    GraderRegistry,
    GraderResult,
    HallucinationGrader,
    PhysicsGrader,
    RedTeamGrader,
    default_registry,
)
from huginn.evaluation.judge import BlindArena, JudgeEvaluator, JudgeResult
from huginn.evaluation.matworld_bench import (
    BenchResult,
    BenchTask,
    CATEGORIES,
    MatWorldBench,
)

__all__ = [
    "evaluate",
    "ArenaRecord",
    "ArenaStore",
    "BlindArena",
    "JudgeEvaluator",
    "JudgeResult",
    # grader (实现在 validation 层, 这里 re-export)
    "GraderResult",
    "GraderRegistry",
    "PhysicsGrader",
    "DimensionalGrader",
    "RedTeamGrader",
    "HallucinationGrader",
    "BenchGrader",
    "default_registry",
    # matworld bench
    "CATEGORIES",
    "BenchTask",
    "BenchResult",
    "MatWorldBench",
]
