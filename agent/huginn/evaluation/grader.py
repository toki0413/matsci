"""Shim — re-exports from huginn.validation.grader.

统一 Grader 接口的实现在 validation 层, 这里只是保持
evaluation 包的导入路径向后兼容.
"""
from huginn.validation.grader import (
    BenchGrader,
    DimensionalGrader,
    GraderRegistry,
    GraderResult,
    HallucinationGrader,
    PhysicsGrader,
    RedTeamGrader,
    default_registry,
)

__all__ = [
    "GraderResult",
    "PhysicsGrader",
    "DimensionalGrader",
    "RedTeamGrader",
    "HallucinationGrader",
    "BenchGrader",
    "GraderRegistry",
    "default_registry",
]
