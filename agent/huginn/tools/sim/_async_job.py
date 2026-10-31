"""共享异步作业提交 — vasp / lammps 的 submit_job 共用同一套后台执行语义.

DRY: 两个工具曾各自复制一份 ``job_entry`` 构造 + ``create_task`` + 无事件循环退化 +
完成回写. 任一处改动 (例如取消/掉电语义) 都会让两种计算工具对同一 agent 表现出
不同的作业行为, 因此收敛为单一实现.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any


async def submit_background(
    prefix: str,
    *,
    compute_action: str,
    working_dir: str | None,
    jobs: dict[str, Any],
    run: Callable[[], Awaitable[Any]],
) -> str:
    """建 job_entry、排后台任务、返回 job_id.

    ``run`` 必须返回带 ``success`` / ``data`` / ``error`` 的结果对象 (ToolResult).
    没有运行中的事件循环时 (同步路径调用) 退化为就地 await —— 此时 "异步" 实际
    阻塞, 但至少功能正确, 与两个工具此前的兜底一致.
    """
    job_id = f"{prefix}-{uuid.uuid4().hex[:12]}"
    job_entry: dict[str, Any] = {
        "status": "running",
        "task": None,
        "result": None,
        "error": None,
        "compute_action": compute_action,
        "working_dir": working_dir,
        "started_at": time.time(),
        "finished_at": None,
    }
    jobs[job_id] = job_entry

    async def _run_in_background() -> None:
        try:
            result = await run()
            job_entry["result"] = result.data if result.success else None
            job_entry["error"] = result.error
            job_entry["status"] = "done" if result.success else "failed"
        except Exception as exc:
            job_entry["error"] = str(exc)
            job_entry["status"] = "failed"
        finally:
            job_entry["finished_at"] = time.time()

    try:
        job_entry["task"] = asyncio.create_task(_run_in_background())
    except RuntimeError:
        # 没有运行中的事件循环 (比如同步路径调用), 退化为同步执行.
        await _run_in_background()
    return job_id
