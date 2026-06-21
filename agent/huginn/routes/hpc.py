"""HPC cluster endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from huginn.hpc.client import HPCClient, HPCConfig

router = APIRouter(tags=["hpc"])


@router.post("/hpc/test")
async def hpc_test_connection(params: dict[str, Any]) -> dict[str, Any]:
    """Test SSH connection to an HPC cluster."""
    cfg = HPCConfig(
        host=params.get("host", ""),
        username=params.get("username", ""),
        scheduler=params.get("scheduler", "slurm"),
        key_path=params.get("key_path"),
        port=params.get("port", 22),
    )

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
    cfg = HPCConfig(
        host=params.get("host", ""),
        username=params.get("username", ""),
        scheduler=params.get("scheduler", "slurm"),
        key_path=params.get("key_path"),
        remote_work_dir=params.get("remote_work_dir", "~/huginn_jobs"),
    )

    if not cfg.host or not cfg.username:
        return {"success": False, "error": "host and username are required"}

    try:
        with HPCClient(cfg) as client:
            script = client.generate_job_script(
                command=params.get("command", "echo 'Hello HPC'"),
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
            return {"success": True, "job_id": job_id, "host": cfg.host}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/hpc/status")
async def hpc_status(params: dict[str, Any]) -> dict[str, Any]:
    """Poll status of a remote HPC job."""
    cfg = HPCConfig(
        host=params.get("host", ""),
        username=params.get("username", ""),
        scheduler=params.get("scheduler", "slurm"),
        key_path=params.get("key_path"),
    )

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
