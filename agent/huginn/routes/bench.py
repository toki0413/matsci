"""Benchmark and self-evolution endpoints."""

from __future__ import annotations

import asyncio
import logging
import traceback
from typing import Any

from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter(tags=["bench"])


@router.post("/bench/run")
async def bench_run(params: dict[str, Any]) -> dict[str, Any]:
    """Run the benchmark suite and optionally trigger a self-evolution cycle."""
    from huginn.bench.runner import BenchmarkRunner
    from huginn.evolution.logger import ExecutionLogger

    try:
        categories = params.get("categories")
        evolve = bool(params.get("evolve", False))
        re_ask = bool(params.get("re_ask", False))
        runner = BenchmarkRunner(logger=ExecutionLogger())
        # runner.run() 内部用 asyncio.run(_agent_chat), 不能在已有 event loop 里调.
        # 放线程里跑, 线程内无 loop, asyncio.run 正常工作.
        report = await asyncio.to_thread(
            runner.run, evolve=evolve, categories=categories, re_ask=re_ask
        )
        return {
            "success": True,
            "report": {
                "run_id": report.run_id,
                "total": report.total,
                "passed": report.passed,
                "failed": report.failed,
                "skipped": report.skipped,
                "unreliable": report.unreliable,
                "metrics": report.metrics,
                "visual_failure_summary": report.visual_failure_summary,
                "results": [
                    {
                        "task_id": r.task_id,
                        "category": r.category,
                        "passed": r.passed,
                        "reason": r.reason,
                        "exec_time_seconds": r.exec_time_seconds,
                        "unreliable": r.unreliable,
                        "visual_capability_tag": r.visual_capability_tag,
                    }
                    for r in report.results
                ],
                "evolution_report": report.evolution_report,
            },
        }
    except Exception as e:
        logger.error("unexpected error", exc_info=True)
        return {"success": False, "error": str(e)}


@router.post("/evolve/run")
async def evolve_run(params: dict[str, Any]) -> dict[str, Any]:
    """Run a self-evolution cycle from recent execution logs."""
    from huginn.evolution.engine import EvolutionEngine
    from huginn.evolution.logger import ExecutionLogger

    try:
        logger = ExecutionLogger(persist_dir=params.get("logs_dir"))
        engine = EvolutionEngine(logger=logger)
        report = engine.run_full_evolution_cycle()
        return {"success": True, "report": report}
    except Exception as e:
        logger.error("unexpected error", exc_info=True)
        return {"success": False, "error": str(e)}
