"""HPC cluster endpoints.

支持两种凭据来源:
1. credential_id — 从凭据库取已保存的 SSH 连接 (host/username/password/key_path
   都在库里, 加密存储), 前端只需传一个 id;
2. 内联参数 — 直接在 body 里传 host/username/key_path/password, 走老逻辑。

两者可混用: 传 credential_id 拿基础配置, 再用 body 字段临时覆盖 (如换
remote_work_dir)。这样既支持"选已保存的集群提交", 也兼容旧前端。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from huginn.hpc.client import HPCClient, HPCConfig

router = APIRouter(tags=["hpc"])


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
        return {"success": False, "error": str(e)}


@router.post("/hpc/submit")
async def hpc_submit(params: dict[str, Any]) -> dict[str, Any]:
    """Submit a job to remote HPC."""
    command = params.get("command", "")
    if not isinstance(command, str) or not command.strip():
        return {"success": False, "error": "command is required and must be a non-empty string"}

    cfg, err = _resolve_hpc_config(params)
    if err or cfg is None:
        return {"success": False, "error": err or "无法解析 HPC 配置"}

    if not cfg.host or not cfg.username:
        return {"success": False, "error": "host and username are required"}

    cid = params.get("credential_id")

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
        )
        store.add_or_update(record)

        return {"success": True, "job_id": job_id, "local_id": local_id, "host": cfg.host}
    except ValueError as e:
        return {"success": False, "error": f"Invalid input: {e}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


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
