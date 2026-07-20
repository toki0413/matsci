"""autoloop/engine.py 拆分: dataclass + snapshot 函数.

抽自 engine.py L178-277, 单一职责 = 类型定义 + 持久化.
不包含 AutoloopEngine 类 (留在 engine.py).

ponytail: 不引新依赖, 不改逻辑, 纯 import 抽取.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LoopPhase:
    """A single phase in the autonomous loop."""

    name: str
    status: str = "pending"  # pending | running | completed | failed
    start_time: float | None = None
    end_time: float | None = None
    result: Any = None
    error: str | None = None


@dataclass
class AutoloopResult:
    """Result of a full autonomous loop iteration."""

    run_id: str
    objective: str
    phases: list[LoopPhase]
    success: bool
    report_path: str | None = None
    total_time_seconds: float = 0.0
    trajectory_path: str | None = None
    goal_achieved: bool | None = None
    goal_judgment: dict[str, Any] | None = None
    # 落盘的 provenance JSONL, run 结束后可回放整条 tool chain
    provenance_path: str | None = None
    # Forest 回流: 多树共识的假设图和提示
    merged_graph: Any = None
    speculator_hint: str = ""


def objective_hash(objective: str) -> str:
    """Stable 8-char hash for an autoloop objective — used to dedup result snapshots.

    Same objective string → same hash → same snapshot file. If two objectives
    only differ by whitespace/casing they hash differently; that's fine, we'd
    rather over-store than silently reuse the wrong run.
    """
    return hashlib.md5(objective.encode("utf-8")).hexdigest()[:8]


def _snapshot_dir(workspace: str | Path) -> Path:
    return Path(workspace) / ".huginn" / "autoloop_results"


def save_autoloop_snapshot(
    result: AutoloopResult, workspace: str | Path,
    in_progress: bool = False,
) -> Path | None:
    """Persist a compact JSON snapshot of an AutoloopResult under
    ``<workspace>/.huginn/autoloop_results/<objective_hash>.json``.

    Lets other components (DeliAutoResearch, future CLI subcommands) reuse a
    finished run without re-instantiating AutoloopEngine. Returns the snapshot
    path, or None on failure — callers treat None as "no snapshot, run normally".

    in_progress=True (P15): 写到 ``<objective_hash>.in_progress.json``,
    跟最终 snapshot 区分; save trigger 中途存, run 结束后写最终 snapshot
    覆盖时 in_progress 文件应被调用方清理 (或下次 run 覆盖).
    """
    try:
        snap_dir = _snapshot_dir(workspace)
        snap_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "objective": result.objective,
            "success": result.success,
            "goal_achieved": result.goal_achieved,
            "goal_judgment": result.goal_judgment,
            "report_path": result.report_path,
            "provenance_path": result.provenance_path,
            "trajectory_path": result.trajectory_path,
            "total_time_seconds": result.total_time_seconds,
            "phases_count": len(result.phases),
            "phases_summary": [
                {"name": p.name, "status": p.status} for p in result.phases
            ],
            "saved_at": time.time(),
            "in_progress": in_progress,
        }
        suffix = ".in_progress" if in_progress else ""
        path = snap_dir / f"{objective_hash(result.objective)}{suffix}.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path
    except Exception:
        logger.debug("failed to save autoloop snapshot", exc_info=True)
        return None


def load_autoloop_snapshot(
    workspace: str | Path, objective: str
) -> dict[str, Any] | None:
    """Read a previously saved snapshot for this objective.

    Returns None if the snapshot is missing or unreadable — callers fall back
    to a fresh engine.run_cognitive() in that case.
    """
    path = _snapshot_dir(workspace) / f"{objective_hash(objective)}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("failed to load autoloop snapshot: %s", path, exc_info=True)
        return None
