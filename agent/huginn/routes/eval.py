"""Unified evaluation pipeline — connects MatWorldBench + GraderRegistry + GoalJudge.

This is the "eval → grade → analyze" loop that agents-cli inspired. The
components already existed in Huginn but weren't wired together into a
single callable pipeline.

POST /eval/run      — run benchmark suite, grade results, save trajectory
GET  /eval/history  — list past eval runs
GET  /eval/results/{run_id} — get a specific eval result
POST /eval/analyze  — cluster failures by pattern from past runs
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter(tags=["eval"])

# Where eval results live on disk
_EVAL_DIR = Path.home() / ".huginn" / "eval"


def _eval_dir() -> Path:
    d = _EVAL_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


@router.post("/eval/run")
async def eval_run(params: dict[str, Any]) -> dict[str, Any]:
    """Run the full evaluation pipeline.

    Steps:
      1. Run MatWorldBench tasks through the agent
      2. Grade each result with GraderRegistry
      3. Run GoalJudge on the overall performance
      4. Save trajectory + eval results to ~/.huginn/eval/{run_id}.json
      5. Return structured report

    Optional params:
      categories: list[str] — filter bench tasks by category
      task_ids: list[str] — run specific tasks only
      goal: str — custom objective for GoalJudge (default: benchmark summary)
    """
    from huginn.evaluation.matworld_bench import MatWorldBench
    from huginn.validation.grader import default_registry
    from huginn.telemetry import TelemetryCollector, save_trajectory

    run_id = f"eval_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    collector = TelemetryCollector()

    # Filter tasks if requested
    bench = MatWorldBench()
    categories = params.get("categories")
    task_ids = params.get("task_ids")
    if task_ids:
        bench.tasks = [t for t in bench.tasks if t.id in task_ids]
    elif categories:
        bench.tasks = [t for t in bench.tasks if t.category in categories]

    if not bench.tasks:
        return {"success": False, "error": "No matching benchmark tasks"}

    # Try to get the agent for live evaluation; fall back to dry-run
    agent = None
    try:
        from huginn.server_core import get_agent
        agent = await get_agent()
    except Exception:
        logger.info("agent not available, eval will run in dry mode")

    # Run each benchmark task
    results: list[dict[str, Any]] = []
    with collector.span("eval_run", run_id=run_id, task_count=len(bench.tasks)):
        for task in bench.tasks:
            task_start = time.time()
            agent_output: dict[str, Any] = {}

            with collector.span("eval_task", task_id=task.id, category=task.category):
                if agent is not None:
                    try:
                        # Use the agent's astream to get a response
                        response_text = ""
                        async for chunk in agent.astream(
                            task.prompt, thread_id=f"eval_{run_id}"
                        ):
                            if isinstance(chunk, dict):
                                response_text += chunk.get("content", "") or ""
                            elif isinstance(chunk, str):
                                response_text += chunk

                        # Try to parse structured output from the response
                        agent_output = _parse_agent_output(response_text)
                        agent_output["_raw_response"] = response_text[:500]
                    except Exception as exc:
                        agent_output = {"__error__": str(exc)}
                        logger.warning("eval task %s failed: %s", task.id, exc)
                else:
                    agent_output = {"__skipped__": "agent not available"}

            # Evaluate against expected results
            bench_result = bench.evaluate(task.id, agent_output)

            # Grade with GraderRegistry
            grader = default_registry()
            grader_results = grader.evaluate_all({
                "output": agent_output,
                "task": {
                    "id": task.id,
                    "category": task.category,
                    "expected": task.expected_result,
                },
            })

            results.append({
                "task_id": task.id,
                "category": task.category,
                "prompt": task.prompt,
                "passed": bench_result.passed,
                "score": bench_result.score,
                "details": bench_result.details,
                "grader_scores": [
                    {"name": g.name, "score": g.score, "passed": g.passed, "message": g.message}
                    for g in grader_results
                ],
                "exec_time_seconds": round(time.time() - task_start, 2),
            })

    # Aggregate
    n_pass = sum(1 for r in results if r["passed"])
    n_total = len(results)
    avg_grader = 0.0
    grader_count = 0
    for r in results:
        for g in r["grader_scores"]:
            avg_grader += g["score"]
            grader_count += 1
    avg_grader = round(avg_grader / grader_count, 4) if grader_count else 0.0

    # GoalJudge — overall objective assessment
    goal_judge_result = None
    goal = params.get("goal", f"Pass {n_total} materials science benchmark tasks")
    try:
        from huginn.evaluation.goal_judge import GoalJudge

        # Use verification model if available, else pass None for rule-based
        judge_llm = None
        if agent is not None and getattr(agent, "model_router", None):
            try:
                judge_llm = agent.model_router.select_verification()
            except Exception:
                pass

        judge = GoalJudge(llm=judge_llm)
        final_output = f"Benchmark pass rate: {n_pass}/{n_total} ({n_pass/n_total*100:.1f}%)\n"
        final_output += "\n".join(
            f"  {r['task_id']}: {'PASS' if r['passed'] else 'FAIL'} (score={r['score']})"
            for r in results
        )
        goal_judge_result = judge.judge(
            objective=goal,
            trajectory=None,
            final_output=final_output,
        )
    except Exception as exc:
        logger.warning("GoalJudge failed: %s", exc)
        goal_judge_result = {"error": str(exc)}

    # Save trajectory
    traj_path = _eval_dir() / f"{run_id}_trajectory.json"
    try:
        save_trajectory(collector, traj_path, metadata={"run_id": run_id, "type": "eval"})
    except Exception:
        logger.debug("trajectory save failed", exc_info=True)

    # Build and persist eval report
    report = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "categories": list({r["category"] for r in results}),
        "total": n_total,
        "passed": n_pass,
        "failed": n_total - n_pass,
        "pass_rate": round(n_pass / n_total, 4) if n_total else 0.0,
        "avg_grader_score": avg_grader,
        "goal_judge": goal_judge_result,
        "results": results,
        "trajectory_path": str(traj_path),
    }

    report_path = _eval_dir() / f"{run_id}.json"
    try:
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("eval report save failed: %s", exc)

    return {"success": True, "report": report}


@router.get("/eval/history")
async def eval_history(limit: int = 20) -> dict[str, Any]:
    """List past eval runs, newest first."""
    runs: list[dict[str, Any]] = []
    try:
        for p in sorted(_eval_dir().glob("eval_*.json"), reverse=True)[:limit]:
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                runs.append({
                    "run_id": data.get("run_id", p.stem),
                    "timestamp": data.get("timestamp", ""),
                    "total": data.get("total", 0),
                    "passed": data.get("passed", 0),
                    "pass_rate": data.get("pass_rate", 0.0),
                    "avg_grader_score": data.get("avg_grader_score", 0.0),
                })
            except Exception:
                continue
    except Exception:
        pass
    return {"runs": runs, "count": len(runs)}


@router.get("/eval/results/{run_id}")
async def eval_results(run_id: str) -> dict[str, Any]:
    """Get full results for a specific eval run."""
    # Sanitize: only allow alphanumeric + underscore
    safe_id = "".join(c for c in run_id if c.isalnum() or c == "_")
    path = _eval_dir() / f"{safe_id}.json"
    if not path.exists():
        return {"success": False, "error": f"eval run '{safe_id}' not found"}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {"success": True, "report": data}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@router.post("/eval/analyze")
async def eval_analyze(params: dict[str, Any]) -> dict[str, Any]:
    """Analyze failure patterns from past eval runs.

    Clusters failures by category and grader, showing which
    dimensions are the weakest.
    """
    limit = int(params.get("limit", 10))
    runs: list[dict[str, Any]] = []
    try:
        for p in sorted(_eval_dir().glob("eval_*.json"), reverse=True)[:limit]:
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                runs.append(data)
            except Exception:
                continue
    except Exception:
        pass

    if not runs:
        return {"success": True, "analysis": {"message": "no past eval runs found"}}

    # Aggregate failures by category and task
    category_stats: dict[str, dict[str, int]] = {}
    task_stats: dict[str, dict[str, int]] = {}
    grader_stats: dict[str, dict[str, int]] = {}

    for run in runs:
        for r in run.get("results", []):
            cat = r.get("category", "unknown")
            tid = r.get("task_id", "unknown")
            passed = r.get("passed", False)

            for target, key in [(category_stats, cat), (task_stats, tid)]:
                if key not in target:
                    target[key] = {"total": 0, "passed": 0, "failed": 0}
                target[key]["total"] += 1
                target[key]["passed" if passed else "failed"] += 1

            for g in r.get("grader_scores", []):
                gname = g.get("name", "unknown")
                if gname not in grader_stats:
                    grader_stats[gname] = {"total": 0, "passed": 0, "failed": 0}
                grader_stats[gname]["total"] += 1
                if g.get("passed", False):
                    grader_stats[gname]["passed"] += 1
                else:
                    grader_stats[gname]["failed"] += 1

    # Find weakest categories
    weak_categories = sorted(
        [
            {
                "category": k,
                "pass_rate": round(v["passed"] / v["total"], 4) if v["total"] else 0.0,
                **v,
            }
            for k, v in category_stats.items()
        ],
        key=lambda x: x["pass_rate"],
    )

    weak_tasks = sorted(
        [
            {
                "task_id": k,
                "pass_rate": round(v["passed"] / v["total"], 4) if v["total"] else 0.0,
                **v,
            }
            for k, v in task_stats.items()
        ],
        key=lambda x: x["pass_rate"],
    )

    return {
        "success": True,
        "analysis": {
            "runs_analyzed": len(runs),
            "total_tasks": sum(v["total"] for v in task_stats.values()),
            "category_breakdown": weak_categories,
            "task_breakdown": weak_tasks,
            "grader_breakdown": [
                {
                    "grader": k,
                    "pass_rate": round(v["passed"] / v["total"], 4) if v["total"] else 0.0,
                    **v,
                }
                for k, v in sorted(grader_stats.items())
            ],
        },
    }


def _parse_agent_output(text: str) -> dict[str, Any]:
    """Best-effort extraction of structured data from agent text response.

    Looks for JSON blocks or key=value patterns in the response.
    """
    import re

    # Try to find a JSON block
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except Exception:
            pass

    # Try bare JSON
    json_match = re.search(r"\{[^{}]*\}", text)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except Exception:
            pass

    # Try key: value patterns (e.g., "band_gap_eV: 1.12")
    result: dict[str, Any] = {}
    for m in re.finditer(r"(\w+)\s*[:=]\s*([\d.eE+-]+)", text):
        key, val = m.group(1), m.group(2)
        try:
            result[key] = float(val)
        except ValueError:
            result[key] = val
    return result
