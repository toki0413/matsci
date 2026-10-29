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
from typing import Any, TypedDict

from huginn.autoloop.plan_store import _file_lock
from huginn.utils.common import atomic_write_json
from huginn.utils.runtime import HUGINN_DIR_NAME

logger = logging.getLogger(__name__)


class CognitiveState(TypedDict, total=False):
    """run_cognitive 的 phase 间中间结果 (cog dict).

    ``total=False``: 键随 phase 推进逐步填充, 未到达的阶段键缺失是正常状态
    (``_rc_new_cog`` 只预置前 8 个). 各读写点统一用此结构 (C1), 避免
    ``cog: dict`` 让拼错的键静默通过.
    """

    context: dict[str, Any]
    hypothesis: str | None
    plan: dict[str, Any] | None
    execution_result: dict[str, Any] | None
    validation: dict[str, Any] | str | None
    current_hyp_id: str | None
    phases: list[Any]
    completed_steps: int
    last_learn_summary: str
    last_learn_detail: Any
    hypothesis_id: str | None
    structure_cognitive_map: Any
    cmap: Any
    structure: Any


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
    # B6: tool_calls 总数 (从 phases[].result["tool_calls"] 累加).
    # 0 = perceive-only 空转, B6 断言会失败. 让"完成但没干活"可被 CI 拦截.
    tool_calls: int = 0
    # A2: 核心循环钩子 (observe/decide/execute/reflect) 的失败记录.
    # 非空 ⇒ harness 出过错, success 必为 False; 字段保留供事后审计.
    loop_failures: list[dict[str, Any]] = field(default_factory=list)


def objective_hash(objective: str) -> str:
    """Stable 8-char hash for an autoloop objective — used to dedup result snapshots.

    Same objective string → same hash → same snapshot file. If two objectives
    only differ by whitespace/casing they hash differently; that's fine, we'd
    rather over-store than silently reuse the wrong run.
    """
    return hashlib.md5(objective.encode("utf-8"), usedforsecurity=False).hexdigest()[:8]


def _snapshot_dir(workspace: str | Path) -> Path:
    return Path(workspace) / HUGINN_DIR_NAME / "autoloop_results"


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
    时同一 objective 的 in_progress 文件会被一并清掉 (否则崩溃重跑会看到
    上一轮留下的"进行中"标记, 误以为还有活着的 run).

    写入用 ``_file_lock`` + ``atomic_write_json``: 进程崩溃只可能留下 tmp 文件,
    不会留下半截 JSON; 并发 CLI / server 同时写同一 objective 也不会撕裂.
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
            "loop_failures": result.loop_failures,
            "saved_at": time.time(),
            "in_progress": in_progress,
        }
        suffix = ".in_progress" if in_progress else ""
        objective_id = objective_hash(result.objective)
        path = snap_dir / f"{objective_id}{suffix}.json"
        with _file_lock(path):
            atomic_write_json(path, payload, indent=2)
            if not in_progress:
                # 最终快照落盘 = 本 run 已收口, 清掉同 objective 的进行中标记.
                stale = snap_dir / f"{objective_id}.in_progress.json"
                if stale.exists():
                    stale.unlink(missing_ok=True)
        return path
    except Exception:  # 防御: 快照保存失败无碍主流程
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
    except Exception:  # 防御: 快照加载失败仅跳过恢复
        logger.debug("failed to load autoloop snapshot: %s", path, exc_info=True)
        return None
