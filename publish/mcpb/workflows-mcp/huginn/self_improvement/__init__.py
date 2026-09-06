"""Benchmark and self-improvement utilities for HuginnAgent.

.. note:: 定位 (2026-08) — 生产主链路已由下列模块承担, 本包保留为
   参考/测试实现, 避免出现两套并存的自进化逻辑:

   - 评测执行: ``huginn.bench.runner.BenchmarkRunner`` (CLI: ``huginn bench``)
   - 自进化:   ``huginn.evolution.engine.EvolutionEngine`` (``--evolve``)

   本包 ``SelfImprovementLoop`` 等类仅被测试引用, 若未来要接入生产,
   应先与上述两套实现的职责对齐, 而非另起炉灶.
"""

from __future__ import annotations

from huginn.self_improvement.core import (
    BenchmarkCase,
    BenchmarkResult,
    BenchmarkSuite,
    CaseTrialResult,
    MultiTrialResult,
    SelfImprovementLoop,
    keyword_evaluator,
    llm_judge_evaluator,
    numeric_evaluator,
    rubric_evaluator,
)

__all__ = [
    "BenchmarkCase",
    "BenchmarkResult",
    "BenchmarkSuite",
    "CaseTrialResult",
    "MultiTrialResult",
    "SelfImprovementLoop",
    "keyword_evaluator",
    "numeric_evaluator",
    "llm_judge_evaluator",
    "rubric_evaluator",
]
