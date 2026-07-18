"""Task-level metrics — 进度/漂移/重定向/PMK 循环次数/工具健康度.

借鉴 PentAGI 的 Langfuse + Grafana 全链路可观测性栈, 落一份 task 级聚合指标,
供 Grafana 仪表盘 + Reflector 介入信号消费. 单步明细走 audit.jsonl, 这里只存
滚动汇总; 每步 update_metrics 后由调用方调 save_metrics 落盘.

Layout: <workspace>/.huginn/task_metrics.json
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path


@dataclass
class TaskMetrics:
    task_id: str
    total_steps: int = 0
    completed_steps: int = 0
    progress: float = 0.0  # 0.0-1.0
    drift_count: int = 0
    redirect_count: int = 0
    pmk_cycle_count: int = 0
    tool_call_health_avg: float = 1.0  # 0.0-1.0
    checkpoint_count: int = 0
    prospective_fired: int = 0
    last_checkpoint_at: str | None = None  # ISO timestamp
    estimated_remaining: int | None = None  # 分钟
    updated_at: str = ""  # ISO timestamp
    # 跨领域支持: domain_label 标记任务所属领域 (materials/physics/chemistry/medicine/math/...)
    # 用于跨领域统计 + 领域包切换. 不强制 LLM 填, 缺失=unknown.
    # ponytail: 字符串标签而非 enum, 避免维护领域枚举. 升级路径: 领域包注册表.
    domain_label: str = "unknown"


def _atomic_write_json(path: Path, payload: dict) -> None:
    # tmp + rename, 跟 checkpoint.py 一个套路
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, default=str))
        os.replace(tmp, str(path))
    except OSError:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _metrics_path(workspace: Path) -> Path:
    return Path(workspace).resolve() / ".huginn" / "task_metrics.json"


def update_metrics(
    task_metrics: TaskMetrics,
    step_evaluation,
    task_state,
    target_chain_progress: float | None = None,
) -> TaskMetrics:
    """根据本步评估滚动更新 metrics, 返回新对象 (不改原对象).

    step_evaluation / task_state 是任意对象, 用 getattr 防御性访问, 字段缺失走默认.
    """
    on_track = getattr(step_evaluation, "on_track", None)
    deviation = getattr(step_evaluation, "deviation", "") or ""
    pmk_feedback = getattr(step_evaluation, "pmk_feedback", "") or ""
    tool_health = getattr(step_evaluation, "tool_call_health", None)

    completed = task_metrics.completed_steps + 1
    drift = task_metrics.drift_count + (1 if on_track == "false" else 0)
    # ponytail: deviation + pmk_feedback 都非空才算重定向 — 仅 deviation 无反馈
    # 算"发现漂移但还没纠偏", 仅 pmk_feedback 无 deviation 算预防性 PMK 不是纠偏.
    # 升级路径: 用 LLM 判定 deviation 与 pmk_feedback 的语义相关性再计数.
    redirected = task_metrics.redirect_count + (
        1 if (deviation.strip() and pmk_feedback.strip()) else 0
    )
    pmk_cycles = task_metrics.pmk_cycle_count + (1 if pmk_feedback.strip() else 0)

    health_avg = task_metrics.tool_call_health_avg
    if tool_health is not None and hasattr(tool_health, "is_anomalous"):
        if tool_health.is_anomalous():
            # ponytail: 简单指数衰减 *0.9, 不做 EWMA. 升级路径: alpha 可调的加权平均.
            health_avg = max(0.0, health_avg * 0.9)

    progress = task_metrics.progress
    if target_chain_progress is not None:
        progress = float(target_chain_progress)

    # 估算剩余分钟数: 用 task_state.created_at 算平均每步耗时, 线性外推
    # ponytail: 线性外推, 不建模非线性 (后期步骤可能更慢/更快). 升级路径: 近期窗口加权.
    estimated = task_metrics.estimated_remaining
    if completed > 0 and task_metrics.total_steps > 0:
        created_at = getattr(task_state, "created_at", None)
        if created_at is not None:
            try:
                elapsed_sec = datetime.now().timestamp() - float(created_at)
                if elapsed_sec > 0:
                    avg_per_step_min = elapsed_sec / 60.0 / completed
                    remaining_steps = max(0, task_metrics.total_steps - completed)
                    estimated = int(remaining_steps * avg_per_step_min)
            except (TypeError, ValueError):
                pass

    return replace(
        task_metrics,
        completed_steps=completed,
        drift_count=drift,
        redirect_count=redirected,
        pmk_cycle_count=pmk_cycles,
        tool_call_health_avg=health_avg,
        progress=progress,
        estimated_remaining=estimated,
        updated_at=datetime.now().isoformat(),
    )


def save_metrics(task_metrics: TaskMetrics, workspace: Path) -> Path:
    """落盘到 workspace/.huginn/task_metrics.json, 原子写, 返回路径."""
    path = _metrics_path(workspace)
    _atomic_write_json(path, asdict(task_metrics))
    return path


def load_metrics(task_id: str, workspace: Path) -> TaskMetrics | None:
    """从 workspace/.huginn/task_metrics.json 加载. 文件不存在或 task_id 不匹配返回 None."""
    path = _metrics_path(workspace)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("task_id") != task_id:
        return None
    return TaskMetrics(**data)


if __name__ == "__main__":
    import shutil
    import tempfile as _tf
    from types import SimpleNamespace

    ws = Path(_tf.mkdtemp(prefix="huginn_tm_test_")) / "ws"
    ws.mkdir()
    try:
        # 1. 基本更新: completed_steps +1, drift/redirect/pmk 都不增, 原对象不变
        m0 = TaskMetrics(task_id="t1", total_steps=10)
        se_ok = SimpleNamespace(
            on_track="true", deviation="", pmk_feedback="",
            tool_call_health=None,
        )
        m1 = update_metrics(m0, se_ok, task_state=None)
        assert m1.completed_steps == 1, f"completed_steps: {m1.completed_steps}"
        assert m1.drift_count == 0
        assert m1.redirect_count == 0
        assert m1.pmk_cycle_count == 0
        assert m1.tool_call_health_avg == 1.0
        assert m1.updated_at != ""
        assert m0.completed_steps == 0  # 原对象未被改
        print("1. 基本更新 OK")

        # 2. 漂移: on_track="false" → drift_count +1; 无 pmk_feedback 不算重定向
        se_drift = SimpleNamespace(
            on_track="false", deviation="差太远", pmk_feedback="",
            tool_call_health=None,
        )
        m2 = update_metrics(m1, se_drift, task_state=None)
        assert m2.drift_count == 1
        assert m2.redirect_count == 0
        print("2. 漂移计数 OK")

        # 3. 重定向 + PMK 循环: deviation + pmk_feedback 都非空
        se_redir = SimpleNamespace(
            on_track="false", deviation="方向偏了",
            pmk_feedback="建议改用 XX 方法", tool_call_health=None,
        )
        m3 = update_metrics(m2, se_redir, task_state=None)
        assert m3.drift_count == 2
        assert m3.redirect_count == 1
        assert m3.pmk_cycle_count == 1
        print("3. 重定向 + PMK OK")

        # 4. 工具健康度衰减: is_anomalous() True → *0.9, 连续两次 → 0.81
        class _TH:
            def is_anomalous(self) -> bool:
                return True

        se_anom = SimpleNamespace(
            on_track="true", deviation="", pmk_feedback="",
            tool_call_health=_TH(),
        )
        m4 = update_metrics(m3, se_anom, task_state=None)
        assert abs(m4.tool_call_health_avg - 0.9) < 1e-9, m4.tool_call_health_avg
        m4b = update_metrics(m4, se_anom, task_state=None)
        assert abs(m4b.tool_call_health_avg - 0.81) < 1e-9, m4b.tool_call_health_avg
        print("4. 工具健康度衰减 OK")

        # 5. target_chain_progress 覆盖 progress; 不传时保持
        m5 = update_metrics(m4b, se_ok, task_state=None, target_chain_progress=0.42)
        assert abs(m5.progress - 0.42) < 1e-9, m5.progress
        m5b = update_metrics(m5, se_ok, task_state=None)
        assert abs(m5b.progress - 0.42) < 1e-9
        print("5. progress 覆盖 OK")

        # 6. estimated_remaining: 用 task_state.created_at 线性外推
        ts = SimpleNamespace(created_at=datetime.now().timestamp() - 600)  # 10 分钟前
        m6 = TaskMetrics(task_id="t1", total_steps=10)
        m6a = update_metrics(m6, se_ok, task_state=ts)
        m6b = update_metrics(m6a, se_ok, task_state=ts)
        # 2 步 ≈ 10 分钟 → 5 分钟/步, 剩 8 步 ≈ 40 分钟
        assert m6b.estimated_remaining is not None
        assert 25 <= m6b.estimated_remaining <= 80, f"estimated: {m6b.estimated_remaining}"
        # task_state=None 时 estimated_remaining 保持 None
        m6c = update_metrics(TaskMetrics(task_id="t1", total_steps=10), se_ok, task_state=None)
        assert m6c.estimated_remaining is None
        print(f"6. estimated_remaining OK (≈{m6b.estimated_remaining}min)")

        # 7. save → load 往返一致
        path = save_metrics(m6b, ws)
        assert path == ws / ".huginn" / "task_metrics.json"
        assert path.exists()
        loaded = load_metrics("t1", ws)
        assert loaded is not None
        assert loaded.task_id == "t1"
        assert loaded.completed_steps == m6b.completed_steps
        assert loaded.drift_count == m6b.drift_count
        assert loaded.redirect_count == m6b.redirect_count
        assert loaded.pmk_cycle_count == m6b.pmk_cycle_count
        assert abs(loaded.tool_call_health_avg - m6b.tool_call_health_avg) < 1e-9
        assert abs(loaded.progress - m6b.progress) < 1e-9
        assert loaded.estimated_remaining == m6b.estimated_remaining
        assert loaded.updated_at == m6b.updated_at
        # 原子写: tmp 不残留
        assert not (ws / ".huginn" / "task_metrics.json.tmp").exists()
        print("7. save → load 往返 OK")

        # 8. load: 文件不存在 / task_id 不匹配都返回 None
        assert load_metrics("t1", ws / "no_such_dir") is None
        assert load_metrics("other_id", ws) is None
        print("8. None 路径 OK")

        print("ALL CHECKS PASSED")
    finally:
        shutil.rmtree(ws.parent, ignore_errors=True)
