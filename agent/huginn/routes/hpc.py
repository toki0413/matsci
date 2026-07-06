"""HPC cluster endpoints.

支持两种凭据来源:
1. credential_id — 从凭据库取已保存的 SSH 连接 (host/username/password/key_path
   都在库里, 加密存储), 前端只需传一个 id;
2. 内联参数 — 直接在 body 里传 host/username/key_path/password, 走老逻辑。

两者可混用: 传 credential_id 拿基础配置, 再用 body 字段临时覆盖 (如换
remote_work_dir)。这样既支持"选已保存的集群提交", 也兼容旧前端。
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from typing import Any

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from huginn.hpc.client import HPCClient, HPCConfig
from huginn.security.auth import require_admin_key

router = APIRouter(tags=["hpc"], dependencies=[Depends(require_admin_key)])

logger = logging.getLogger(__name__)

# Simple in-memory rate limiter for HPC job submission
_hpc_submit_timestamps: dict[str, list[float]] = defaultdict(list)
_HPC_RATE_LIMIT_PER_MINUTE = 10  # max 10 submissions per minute per user


def _check_hpc_rate_limit(user_id: str = "default") -> bool:
    now = time.time()
    cutoff = now - 60.0
    timestamps = _hpc_submit_timestamps[user_id]
    # Remove old entries
    _hpc_submit_timestamps[user_id] = [t for t in timestamps if t > cutoff]
    if len(_hpc_submit_timestamps[user_id]) >= _HPC_RATE_LIMIT_PER_MINUTE:
        return False
    _hpc_submit_timestamps[user_id].append(now)
    return True


def _resolve_hpc_config(params: dict[str, Any]) -> tuple[HPCConfig | None, str | None]:
    """从请求参数构造 HPCConfig。

    优先用 credential_id 从凭据库加载 (含解密后的 password), 再用 body
    里的字段做临时覆盖。没传 credential_id 就走内联参数老逻辑。

    返回 (config, error); error 非 None 时 config 为 None。
    """
    cid = params.get("credential_id")
    if cid:
        # 延迟导入避免 routes <-> security 的循环依赖
        from huginn.security.credential_store import get_credential_store

        cfg = get_credential_store().to_hpc_config(cid)
        if cfg is None:
            return None, f"credential_id '{cid}' 对应的 SSH 凭据不存在"
        # 允许 body 字段覆盖部分配置 (比如临时换工作目录)
        if params.get("remote_work_dir"):
            cfg.remote_work_dir = params["remote_work_dir"]
        if params.get("scheduler"):
            cfg.scheduler = params["scheduler"]
        return cfg, None

    # 内联参数 — 老逻辑, 现在补上 password 字段 (之前漏了)
    cfg = HPCConfig(
        host=params.get("host", ""),
        username=params.get("username", ""),
        scheduler=params.get("scheduler", "slurm"),
        key_path=params.get("key_path"),
        password=params.get("password"),
        port=params.get("port", 22),
        remote_work_dir=params.get("remote_work_dir", "~/huginn_jobs"),
    )
    return cfg, None


@router.post("/hpc/test")
async def hpc_test_connection(params: dict[str, Any]) -> dict[str, Any]:
    """Test SSH connection to an HPC cluster."""
    cfg, err = _resolve_hpc_config(params)
    if err or cfg is None:
        return {"success": False, "error": err or "无法解析 HPC 配置"}

    if not cfg.host or not cfg.username:
        return {"success": False, "error": "host and username are required"}

    try:
        with HPCClient(cfg) as client:
            stdout, stderr, rc = client._exec("hostname")
            if rc == 0:
                return {
                    "success": True,
                    "hostname": stdout,
                    "scheduler": cfg.scheduler,
                }
            else:
                return {"success": False, "error": stderr or "Connection failed"}
    except Exception as e:
        logger.error("HPC test connection failed: %s", e, exc_info=True)
        return {"success": False, "error": "Connection failed (see server logs)"}


@router.post("/hpc/submit")
async def hpc_submit(params: dict[str, Any]) -> dict[str, Any]:
    """Submit a job to remote HPC.

    支持增强的队列参数: priority / depends_on / dependency_type /
    array_spec / walltime_estimate, 老的调用不传这些字段照常工作。
    """
    if not _check_hpc_rate_limit():
        return JSONResponse(
            status_code=429,
            content={"success": False, "error": "Rate limit exceeded: max 10 HPC submissions per minute"},
        )

    command = params.get("command", "")
    if not isinstance(command, str) or not command.strip():
        return {"success": False, "error": "command is required and must be a non-empty string"}

    cfg, err = _resolve_hpc_config(params)
    if err or cfg is None:
        return {"success": False, "error": err or "无法解析 HPC 配置"}

    if not cfg.host or not cfg.username:
        return {"success": False, "error": "host and username are required"}

    cid = params.get("credential_id")

    # 取队列增强参数, 没传就用默认值
    priority = params.get("priority", "normal")
    depends_on = params.get("depends_on") or []
    dependency_type = params.get("dependency_type", "afterok")
    array_spec = params.get("array_spec")
    walltime_estimate = params.get("walltime_estimate")

    # 校验优先级, 防止传进来乱七八糟的值
    if priority not in ("low", "normal", "high", "urgent"):
        return {"success": False, "error": f"无效的优先级: {priority}"}

    try:
        with HPCClient(cfg) as client:
            script = client.generate_job_script(
                command=command,
                job_name=params.get("job_name", "huginn_job"),
                walltime=params.get("walltime", "01:00:00"),
                nodes=params.get("nodes", 1),
                ntasks_per_node=params.get("ntasks_per_node", 4),
                queue=params.get("queue"),
                modules=params.get("modules", []),
                env_vars=params.get("env_vars", {}),
                priority=priority,
                depends_on=depends_on,
                dependency_type=dependency_type,
                array_spec=array_spec,
            )
            job_id = client.submit_job(
                script, job_name=params.get("job_name", "huginn_job")
            )

        # Persist to the job store so we can list/cancel/refresh later
        import os
        import shlex
        import time
        import uuid
        from pathlib import Path

        from huginn.execution.remote_job_store import RemoteJobRecord, RemoteJobStore

        workspace = Path(os.environ.get("HUGINN_WORKSPACE", "."))
        store = RemoteJobStore(workspace=workspace)
        local_id = str(uuid.uuid4())[:8]
        record = RemoteJobRecord(
            local_id=local_id,
            scheduler_id=str(job_id),
            command=shlex.split(command),
            cwd=cfg.remote_work_dir,
            credential_id=cid,
            queue=params.get("queue"),
            status="PENDING",
            submitted_at=time.time(),
            priority=priority,
            depends_on=list(depends_on),
            dependency_type=dependency_type,
            array_spec=array_spec,
            walltime_estimate=walltime_estimate,
        )
        store.add_or_update(record)

        return {
            "success": True,
            "job_id": job_id,
            "local_id": local_id,
            "host": cfg.host,
            "priority": priority,
            "array_spec": array_spec,
        }
    except ValueError as e:
        return {"success": False, "error": f"Invalid input: {e}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── 队列增强: 优先级 / 依赖 / 数组 / walltime 估算 ────────────────
#
# 这几个端点给前端做"高级提交"向导用: 列优先级档位、估算 walltime、
# 查询作业依赖图。submit 端点本身已经支持这些字段, 这里只是辅助。


@router.get("/hpc/queue/priorities")
async def hpc_list_priorities() -> dict[str, Any]:
    """返回支持的优先级档位及说明, 给前端下拉框用。"""
    return {
        "success": True,
        "priorities": [
            {"value": "low", "label": "低", "description": "空闲时才跑"},
            {"value": "normal", "label": "普通", "description": "默认优先级"},
            {"value": "high", "label": "高", "description": "优先调度"},
            {"value": "urgent", "label": "紧急", "description": "最高优先级, 抢占式"},
        ],
    }


@router.post("/hpc/estimate-walltime")
async def hpc_estimate_walltime(params: dict[str, Any]) -> dict[str, Any]:
    """根据命令类型和历史经验粗略估算 walltime。

    估算逻辑比较朴素: 按命令里的关键词 (vasp / lammps / qe / gaussian 等)
    匹配典型耗时档位, 用户可覆盖。真实耗时还得看具体规模, 这里只给个起点。
    """
    command = (params.get("command") or "").lower()
    nodes = params.get("nodes", 1)
    ntasks = params.get("ntasks_per_node", 4)

    # 不同计算软件的典型单步耗时经验值 (小时), 真实场景差异很大
    benchmarks = [
        ("vasp", 2.0, "VASP 结构优化/SCF, 中等体系"),
        ("lammps", 1.5, "LAMMPS 分子动力学"),
        ("pw.x", 3.0, "Quantum ESPRESSO SCF"),
        ("g16", 4.0, "Gaussian 计算"),
        ("cp2k", 2.5, "CP2K 计算"),
        ("abinit", 2.0, "ABINIT 计算"),
    ]

    estimate_hours = 1.0
    matched = "默认估算 (通用任务)"
    for kw, hours, desc in benchmarks:
        if kw in command:
            estimate_hours = hours
            matched = desc
            break

    # 节点数越多, 单作业分摊的时间略降; 这里简单按 1/sqrt(n) 缩放
    if nodes > 1:
        estimate_hours = estimate_hours / (nodes ** 0.5)

    total_cores = nodes * ntasks
    # 转成 HH:MM:SS
    total_seconds = int(estimate_hours * 3600)
    hh, rem = divmod(total_seconds, 3600)
    mm, ss = divmod(rem, 60)
    walltime = f"{hh:02d}:{mm:02d}:{ss:02d}"

    return {
        "success": True,
        "walltime": walltime,
        "estimate_hours": round(estimate_hours, 2),
        "matched_type": matched,
        "total_cores": total_cores,
        "nodes": nodes,
        "ntasks_per_node": ntasks,
    }


@router.get("/hpc/jobs/{local_id}/dependencies")
async def hpc_job_dependencies(local_id: str) -> dict[str, Any]:
    """查询某作业的依赖关系 (前置 + 后继), 画依赖图用。"""
    store = _get_store()
    record = store.get(local_id)
    if record is None:
        return {"success": False, "error": f"job '{local_id}' not found"}

    # 找出所有依赖本作业的后续作业
    all_jobs = store.list_jobs()
    dependents = [
        {"local_id": j.local_id, "scheduler_id": j.scheduler_id, "status": j.status}
        for j in all_jobs
        if record.scheduler_id in j.depends_on
    ]

    return {
        "success": True,
        "job": record.to_dict(),
        "depends_on": [
            {"scheduler_id": sid} for sid in record.depends_on
        ],
        "dependents": dependents,
    }


# ── Tracked job management ───────────────────────────────────────
#
# These endpoints operate on the RemoteJobStore rather than going through
# RemoteExecutor. The store is JSON-backed and survives restarts, so jobs
# submitted via /hpc/submit can be listed, refreshed, and cancelled even
# after the agent process restarts — as long as credential_id was saved.


def _get_store():
    """Build a RemoteJobStore rooted at HUGINN_WORKSPACE."""
    import os
    from pathlib import Path

    from huginn.execution.remote_job_store import RemoteJobStore

    workspace = Path(os.environ.get("HUGINN_WORKSPACE", "."))
    return RemoteJobStore(workspace=workspace)


@router.get("/hpc/jobs")
async def hpc_list_jobs(credential_id: str | None = None) -> dict[str, Any]:
    """List all tracked remote jobs from the job store."""
    store = _get_store()
    jobs = store.list_jobs()

    if credential_id:
        jobs = [j for j in jobs if j.credential_id == credential_id]

    return {
        "success": True,
        "jobs": [j.to_dict() for j in jobs],
        "count": len(jobs),
    }


@router.get("/hpc/jobs/{local_id}")
async def hpc_get_job(local_id: str) -> dict[str, Any]:
    """Get details of a specific remote job."""
    store = _get_store()
    record = store.get(local_id)
    if record is None:
        return {"success": False, "error": f"job '{local_id}' not found"}
    return {"success": True, "job": record.to_dict()}


@router.post("/hpc/jobs/{local_id}/refresh")
async def hpc_refresh_job(
    local_id: str, params: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Poll the scheduler for the latest status of a tracked job."""
    import time

    store = _get_store()
    record = store.get(local_id)
    if record is None:
        return {"success": False, "error": f"job '{local_id}' not found"}

    # Prefer the credential_id stored on the record; fall back to inline params
    params = params or {}
    cfg, err = _resolve_hpc_config({
        "credential_id": record.credential_id or params.get("credential_id"),
        **params,
    })
    if err or cfg is None:
        return {"success": False, "error": err or "cannot resolve HPC config"}

    try:
        with HPCClient(cfg) as client:
            status = client.poll_status(record.scheduler_id)
            record.status = status.state
            record.exit_code = status.exit_code
            record.message = status.message
            if status.state in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"):
                record.completed_at = time.time()
            store.add_or_update(record)
            return {"success": True, "job": record.to_dict()}
    except Exception as e:
        return {"success": False, "error": type(e).__name__}


@router.post("/hpc/jobs/{local_id}/cancel")
async def hpc_cancel_job(
    local_id: str, params: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Cancel a tracked job on the remote scheduler."""
    import shlex
    import time

    store = _get_store()
    record = store.get(local_id)
    if record is None:
        return {"success": False, "error": f"job '{local_id}' not found"}

    params = params or {}
    cfg, err = _resolve_hpc_config({
        "credential_id": record.credential_id or params.get("credential_id"),
        **params,
    })
    if err or cfg is None:
        return {"success": False, "error": err or "cannot resolve HPC config"}

    try:
        with HPCClient(cfg) as client:
            if cfg.scheduler == "slurm":
                client._exec(f"scancel {shlex.quote(record.scheduler_id)}")
            else:
                client._exec(f"qdel {shlex.quote(record.scheduler_id)}")
            record.status = "CANCELLED"
            record.completed_at = time.time()
            store.add_or_update(record)
            return {"success": True, "job": record.to_dict()}
    except Exception as e:
        return {"success": False, "error": type(e).__name__}


@router.post("/hpc/status")
async def hpc_status(params: dict[str, Any]) -> dict[str, Any]:
    """Poll status of a remote HPC job."""
    cfg, err = _resolve_hpc_config(params)
    if err or cfg is None:
        return {"success": False, "error": err or "无法解析 HPC 配置"}

    job_id = params.get("job_id")
    if not job_id:
        return {"success": False, "error": "job_id is required"}

    try:
        with HPCClient(cfg) as client:
            status = client.poll_status(job_id)
            return {
                "success": True,
                "job_id": status.job_id,
                "state": status.state,
                "exit_code": status.exit_code,
                "runtime": status.runtime,
                "message": status.message,
            }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── 实时输出流: WebSocket 尾随作业日志 ────────────────────────────
#
# 类似 MobaXterm 的终端, 连上以后实时推 slurm-%j.out 的内容。
# 客户端断开或作业结束时自动收尾, 不死连。


def _job_output_path(scheduler: str, job_id: str, work_dir: str) -> str:
    """拼出作业输出文件的远程路径。

    SLURM 默认 slurm-{jobid}.out, PBS 默认 pbs-{jobid}.out,
    都落在 remote_work_dir 下。数组作业的子任务日志文件名带 _{index}。
    """
    if scheduler == "pbs":
        return f"{work_dir}/pbs-{job_id}.out"
    return f"{work_dir}/slurm-{job_id}.out"


@router.websocket("/ws/hpc/jobs/{local_id}/output")
async def hpc_job_output_stream(websocket: WebSocket, local_id: str):
    """实时尾随作业 stdout 日志, 通过 WebSocket 推给前端。

    连接建立后先发一段历史输出, 然后持续 tail -f 新增内容,
    直到客户端断开或作业结束。
    """
    # WebSocket 路由不走 router 级依赖, 手动鉴权
    from huginn.security.auth import require_api_key

    try:
        require_api_key(request=None, websocket=websocket)
    except Exception:
        await websocket.close(code=4001, reason="Unauthorized")
        return

    await websocket.accept()

    store = _get_store()
    record = store.get(local_id)
    if record is None:
        await websocket.send_json({"type": "error", "error": f"job '{local_id}' not found"})
        await websocket.close()
        return

    cfg, err = _resolve_hpc_config({"credential_id": record.credential_id})
    if err or cfg is None:
        await websocket.send_json({"type": "error", "error": "无法解析 HPC 配置"})
        await websocket.close()
        return

    output_file = _job_output_path(cfg.scheduler, record.scheduler_id, cfg.remote_work_dir)

    # 把阻塞的 SSH tail 放到线程里跑, 主协程负责转发
    try:
        await websocket.send_json({
            "type": "info",
            "job_id": record.scheduler_id,
            "local_id": local_id,
            "output_file": output_file,
            "status": record.status,
        })

        # 用一个队列衔接线程 (读 SSH) 和协程 (写 WebSocket)
        line_queue: asyncio.Queue = asyncio.Queue()
        # 在协程里拿到事件循环引用, 传给线程用; 线程里直接 get_event_loop 会炸
        loop = asyncio.get_running_loop()

        def _tail_thread():
            """在线程里连 SSH, tail -f 输出文件, 每行塞进队列。"""
            import shlex

            try:
                with HPCClient(cfg) as client:
                    # 先确认文件在不在, 不在就等它出现
                    stdout, stderr, rc = client._exec(
                        ["test", "-f", output_file, "&&", "echo", "EXISTS"]
                    )
                    if "EXISTS" not in stdout:
                        # 文件还没生成, 轮询等
                        for _ in range(60):
                            import time as _t
                            _t.sleep(2)
                            stdout, _, _ = client._exec(
                                ["test", "-f", output_file, "&&", "echo", "EXISTS"]
                            )
                            if "EXISTS" in stdout:
                                break
                        else:
                            asyncio.run_coroutine_threadsafe(
                                line_queue.put(("error", "输出文件未生成, 等待超时")),
                                loop,
                            )
                            return

                    # tail -f 实时跟, 用 exec_command 的 stdout 流式读
                    client._ensure_connected()
                    cmd = f"tail -n +1 -f {shlex.quote(output_file)}"
                    _, stdout, _ = client._ssh.exec_command(cmd, get_pty=False)

                    for raw_line in iter(stdout.readline, None):
                        if raw_line == "" or raw_line is None:
                            break
                        try:
                            asyncio.run_coroutine_threadsafe(
                                line_queue.put(("line", raw_line)), loop
                            )
                        except RuntimeError:
                            break  # 事件循环已关
            except Exception as exc:
                try:
                    asyncio.run_coroutine_threadsafe(
                        line_queue.put(("error", str(exc))), loop
                    )
                except RuntimeError:
                    pass

        import threading

        tail_thread = threading.Thread(target=_tail_thread, daemon=True)
        tail_thread.start()

        # 主协程: 从队列取行, 推给 WebSocket
        terminal_states = ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT")
        idle_count = 0
        while True:
            try:
                # 1 秒超时, 没数据就检查作业状态
                kind, content = await asyncio.wait_for(line_queue.get(), timeout=1.0)
                idle_count = 0
                if kind == "line":
                    await websocket.send_json({"type": "output", "line": content})
                elif kind == "error":
                    await websocket.send_json({"type": "error", "error": content})
                    break
            except asyncio.TimeoutError:
                idle_count += 1
                # 每隔 ~15 秒轮一次作业状态, 看是不是结束了
                if idle_count % 15 == 0:
                    try:
                        with HPCClient(cfg) as client:
                            status = client.poll_status(record.scheduler_id)
                        await websocket.send_json({
                            "type": "status",
                            "state": status.state,
                            "exit_code": status.exit_code,
                        })
                        if status.state in terminal_states:
                            await websocket.send_json({"type": "done"})
                            break
                    except Exception:
                        pass
            except WebSocketDisconnect:
                break
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.error("作业输出流异常: %s", exc, exc_info=True)
        try:
            await websocket.send_json({"type": "error", "error": str(exc)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
