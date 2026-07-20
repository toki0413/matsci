"""后台任务命令 — 用线程池跑长耗时的 objective, 不阻塞 chat。

用法:
    huginn bg start "筛选100种钙钛矿材料"
    huginn bg list
    huginn bg status <task_id>
    huginn bg stop <task_id>
    huginn bg result <task_id>

任务状态写入 ~/.huginn/background_tasks.json, 结果写入 ~/.huginn/bg_results/<id>.txt。
跨进程可读, 但 stop 只能取消当前进程内还在跑的任务。
"""

from __future__ import annotations

import json
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import click
from rich.table import Table

from huginn.cli.context import CliContext
from huginn.utils.common import now_iso

# 状态文件默认放 ~/.huginn/, 跟 sessions 的 sqlite 一个目录
_DEFAULT_STATE_FILE = Path.home() / ".huginn" / "background_tasks.json"
_DEFAULT_RESULTS_DIR = Path.home() / ".huginn" / "bg_results"

# 线程池大小 4 够用了, 多了反而抢资源
_MAX_WORKERS = 4

# 进程级单例, 整个 CLI 生命周期共享一个 executor
_executor: ThreadPoolExecutor | None = None
_executor_lock = threading.Lock()


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        with _executor_lock:
            if _executor is None:
                _executor = ThreadPoolExecutor(
                    max_workers=_MAX_WORKERS, thread_name_prefix="huginn-bg"
                )
    return _executor


def _truncate(text: str, width: int = 60) -> str:
    """截断字符串, 超长加省略号, 表格预览用。"""
    text = text.strip().replace("\n", " ")
    return text if len(text) <= width else text[: width - 3] + "..."


class BackgroundTaskManager:
    """后台任务管理器。

    状态写入 JSON 文件, 跨进程可读; futures 存内存, 只在当前进程有效。
    所以 stop 只能取消本进程提交的任务, 跨进程的 stop 会标记成 stopping,
    实际能否停掉取决于任务线程是否检查状态。
    """

    _instance: "BackgroundTaskManager | None" = None
    _instance_lock = threading.Lock()

    def __init__(
        self,
        state_file: Path | None = None,
        results_dir: Path | None = None,
    ) -> None:
        self._state_file = state_file or _DEFAULT_STATE_FILE
        self._results_dir = results_dir or _DEFAULT_RESULTS_DIR
        # task_id -> Future, 只在内存里, 进程退出就没了
        self._futures: dict[str, Future] = {}
        self._lock = threading.RLock()

    @classmethod
    def get_instance(cls) -> "BackgroundTaskManager":
        """拿全局单例, 第一次调用时创建。"""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # --- 状态文件读写 ---

    def _read_state(self) -> dict[str, dict]:
        if not self._state_file.exists():
            return {}
        try:
            data = json.loads(self._state_file.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}

    def _write_state(self, state: dict[str, dict]) -> None:
        # 原子写: 先写 tmp 再 rename, 防止中途崩溃把状态文件写坏
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._state_file.with_suffix(self._state_file.suffix + ".tmp")
        tmp.write_text(
            json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        tmp.replace(self._state_file)

    def _update_task(self, task_id: str, **fields: Any) -> None:
        with self._lock:
            state = self._read_state()
            task = state.setdefault(task_id, {"task_id": task_id})
            task.update(fields)
            self._write_state(state)

    # --- 公开 API ---

    def start(
        self,
        objective: str,
        agent: Any | None = None,
        agent_factory: Callable[[], Any] | None = None,
    ) -> str:
        """启动后台任务。

        agent 和 agent_factory 二选一:
          - agent: 直接用现成的 agent (chat 里 /bg 走这条路)
          - agent_factory: 在任务线程里延迟建 agent (click 命令走这条路, 避免阻塞主线程)

        返回 task_id。
        """
        task_id = uuid.uuid4().hex[:8]
        result_path = self._results_dir / f"{task_id}.txt"
        result_path.parent.mkdir(parents=True, exist_ok=True)

        self._update_task(
            task_id,
            objective=objective,
            status="running",
            created_at=now_iso(),
            updated_at=now_iso(),
            result_path=str(result_path),
            error=None,
        )

        future = _get_executor().submit(
            self._run_task, task_id, objective, agent, agent_factory, result_path
        )
        with self._lock:
            self._futures[task_id] = future
        return task_id

    def _run_task(
        self,
        task_id: str,
        objective: str,
        agent: Any | None,
        agent_factory: Callable[[], Any] | None,
        result_path: Path,
    ) -> None:
        """任务线程入口, 跑 agent.invoke 并把结果写文件。"""
        try:
            # 没现成 agent 就用 factory 建一个 (在任务线程里建, 不阻塞主线程)
            if agent is None and agent_factory is not None:
                agent = agent_factory()
            if agent is None:
                raise RuntimeError("没有可用的 agent, 请先配置 provider")

            result = agent.invoke(objective)
            content = self._extract_result_text(result)
            result_path.write_text(content, encoding="utf-8")
            self._update_task(
                task_id,
                status="completed",
                updated_at=now_iso(),
                result_preview=_truncate(content, 200),
            )
        except Exception as e:
            self._update_task(
                task_id,
                status="failed",
                error=str(e),
                updated_at=now_iso(),
            )
        finally:
            with self._lock:
                self._futures.pop(task_id, None)

    @staticmethod
    def _extract_result_text(result: Any) -> str:
        """从 agent.invoke 的返回值里挖出可读文本。"""
        if isinstance(result, dict):
            messages = result.get("messages", [])
            if messages:
                last = messages[-1]
                content = getattr(last, "content", None)
                if isinstance(content, str):
                    return content
                if content is None:
                    return ""
                return str(content)
            # 没有 messages 就把整个 dict dump 出来
            return json.dumps(result, indent=2, ensure_ascii=False, default=str)
        return str(result)

    def list_tasks(self) -> list[dict]:
        """返回所有任务, 按创建时间倒序。"""
        state = self._read_state()
        tasks = list(state.values())
        tasks.sort(key=lambda t: t.get("created_at", ""), reverse=True)
        return tasks

    def get_status(self, task_id: str) -> dict | None:
        return self._read_state().get(task_id)

    def stop(self, task_id: str) -> tuple[bool, str]:
        """停止任务。

        返回 (是否已取消, 消息)。
        如果任务还在 future 队列里, 尝试 cancel();
        已经在跑的没法硬中断, 标记成 stopping 让上层提示用户。
        """
        with self._lock:
            future = self._futures.get(task_id)

        if future is not None:
            if future.cancel():
                self._update_task(
                    task_id, status="stopped", updated_at=now_iso()
                )
                with self._lock:
                    self._futures.pop(task_id, None)
                return True, "已取消"
            # 已在跑, cancel 不了
            self._update_task(
                task_id, status="stopping", updated_at=now_iso()
            )
            return False, "任务正在运行, 已标记停止"

        # 跨进程或已结束的任务, 看状态文件
        task = self.get_status(task_id)
        if task is None:
            return False, f"任务 {task_id} 不存在"
        if task.get("status") in ("completed", "failed", "stopped"):
            return False, f"任务已结束 ({task.get('status')})"
        # 标记一下, 实际停不停得看运行方是否检查
        self._update_task(task_id, status="stopping", updated_at=now_iso())
        return False, "已标记停止 (跨进程无法硬中断)"

    def get_result(self, task_id: str) -> str | None:
        """读任务结果文件, 没有返回 None。"""
        task = self.get_status(task_id)
        if task is None:
            return None
        result_path = Path(task.get("result_path", ""))
        if not result_path.exists():
            return None
        try:
            return result_path.read_text(encoding="utf-8")
        except OSError:
            return None


# ── click 命令 ──────────────────────────────────────────────────────


@click.group(name="bg")
def bg() -> None:
    """Manage background tasks."""


@bg.command("start")
@click.argument("objective")
@click.pass_obj
def bg_start(ctx: CliContext, objective: str) -> None:
    """Start a background task with the given objective."""
    from huginn.cli.context import build_agent_from_ctx

    manager = BackgroundTaskManager.get_instance()

    # 在任务线程里延迟建 agent, 主线程立刻返回 task_id
    def _factory() -> Any:
        return build_agent_from_ctx(ctx)

    task_id = manager.start(objective, agent_factory=_factory)
    ctx.console.print(f"[green]✓[/green] 后台任务已启动, id={task_id}")
    ctx.console.print(
        f"[dim]用 `huginn bg status {task_id}` 查看进度, "
        f"`huginn bg result {task_id}` 看结果[/dim]"
    )


@bg.command("list")
@click.pass_obj
def bg_list(ctx: CliContext) -> None:
    """List background tasks."""
    manager = BackgroundTaskManager.get_instance()
    tasks = manager.list_tasks()
    if not tasks:
        ctx.console.print("[yellow]没有后台任务[/yellow]")
        return

    table = Table(title="后台任务")
    table.add_column("ID", style="cyan")
    table.add_column("状态", justify="center")
    table.add_column("目标")
    table.add_column("创建时间")

    status_color = {
        "running": "yellow",
        "completed": "green",
        "failed": "red",
        "stopped": "dim",
        "stopping": "yellow",
    }
    for t in tasks:
        status = t.get("status", "?")
        color = status_color.get(status, "white")
        table.add_row(
            t.get("task_id", ""),
            f"[{color}]{status}[/{color}]",
            _truncate(t.get("objective", "")),
            (t.get("created_at", "") or "")[:19],
        )
    ctx.console.print(table)


@bg.command("status")
@click.argument("task_id")
@click.pass_obj
def bg_status(ctx: CliContext, task_id: str) -> None:
    """Check status of a background task."""
    manager = BackgroundTaskManager.get_instance()
    task = manager.get_status(task_id)
    if task is None:
        ctx.console.print(f"[red]任务 {task_id} 不存在[/red]")
        return
    for k, v in task.items():
        ctx.console.print(f"[cyan]{k}[/cyan]: {v}")


@bg.command("stop")
@click.argument("task_id")
@click.pass_obj
def bg_stop(ctx: CliContext, task_id: str) -> None:
    """Stop a background task."""
    manager = BackgroundTaskManager.get_instance()
    ok, msg = manager.stop(task_id)
    if ok:
        ctx.console.print(f"[green]✓[/green] 任务 {task_id} {msg}")
    else:
        ctx.console.print(f"[yellow]任务 {task_id}: {msg}[/yellow]")


@bg.command("result")
@click.argument("task_id")
@click.pass_obj
def bg_result(ctx: CliContext, task_id: str) -> None:
    """View result of a completed background task."""
    manager = BackgroundTaskManager.get_instance()
    text = manager.get_result(task_id)
    if text is None:
        ctx.console.print(f"[yellow]任务 {task_id} 还没有结果[/yellow]")
        return
    ctx.console.print(text)


__all__ = ["BackgroundTaskManager", "bg"]
