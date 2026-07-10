"""Evaluation — MCDA 决策分析 + 统一 Grader."""

from huginn.evaluation.core import evaluate
from huginn.evaluation.goal_judge import GoalJudge
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
from huginn.evaluation.matworld_bench import (
    BenchResult,
    BenchTask,
    CATEGORIES,
    MatWorldBench,
)

__all__ = [
    "evaluate",
    "GoalJudge",
    "GraderResult",
    "GraderRegistry",
    "PhysicsGrader",
    "DimensionalGrader",
    "RedTeamGrader",
    "HallucinationGrader",
    "BenchGrader",
    "default_registry",
    "CATEGORIES",
    "BenchTask",
    "BenchResult",
    "MatWorldBench",
]
