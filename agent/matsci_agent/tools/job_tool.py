"""HPC job management tool — submit, monitor, and control computational jobs.

Supports both local mock mode and remote HPC submission via SSH.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from matsci_agent.tools.base import MatSciTool
from matsci_agent.types import ToolResult, ToolContext


class JobToolInput(BaseModel):
    action: Literal[
        "submit", "status", "cancel", "list",
        "submit_remote", "poll_remote", "download_remote"
    ] = Field(...)
    
    # Local submission
    script_path: str | None = Field(default=None, description="Path to job submission script")
    queue: Literal["debug", "normal", "gpu", "fat"] = Field(default="normal")
    walltime_hours: int = Field(default=24, ge=1, le=168)
    cores: int = Field(default=4, ge=1)
    memory_gb: int = Field(default=16, ge=1)
    job_id: str | None = Field(default=None)
    
    # Remote HPC
    hpc_host: str | None = Field(default=None, description="HPC hostname (for remote actions)")
    hpc_username: str | None = Field(default=None, description="HPC username")
    hpc_scheduler: Literal["slurm", "pbs"] = Field(default="slurm")
    hpc_key_path: str | None = Field(default=None, description="SSH key path")
    remote_work_dir: str | None = Field(default=None)
    
    # Job script generation (for submit_remote without script_path)
    command: str | None = Field(default=None, description="Command to run on HPC")
    job_name: str | None = Field(default=None)
    modules: list[str] = Field(default_factory=list, description="Modules to load")
    env_vars: dict[str, str] = Field(default_factory=dict)
    
    # Download
    remote_path: str | None = Field(default=None)
    local_path: str | None = Field(default=None)


class JobToolOutput(BaseModel):
    job_id: str | None = None
    status: Literal["submitted", "queued", "running", "completed", "failed", "cancelled", "unknown"] | None = None
    queue_position: int | None = None
    runtime: str | None = None
    output_path: str | None = None
    message: str | None = None
    files: list[str] | None = None


class JobTool(MatSciTool):
    """Submit and manage HPC jobs locally or remotely."""
    
    name = "job_tool"
    description = "Submit, monitor, and cancel computational jobs on HPC clusters (Slurm/PBS). Supports remote SSH submission."
    input_schema = JobToolInput
    
    def is_read_only(self, args: JobToolInput) -> bool:
        return args.action in ["status", "list", "poll_remote"]
    
    def estimate_cost(self, args: JobToolInput) -> dict[str, float] | None:
        if args.action in ["submit", "submit_remote"]:
            return {
                "cpu_hours": args.cores * args.walltime_hours,
                "walltime_hours": args.walltime_hours,
            }
        return None
    
    async def call(self, args: JobToolInput, context: ToolContext) -> ToolResult:
        if args.action == "submit":
            return self._submit_local(args)
        elif args.action == "status":
            return self._status_local(args)
        elif args.action == "cancel":
            return self._cancel_local(args)
        elif args.action == "list":
            return self._list_local(args)
        elif args.action == "submit_remote":
            return await self._submit_remote(args)
        elif args.action == "poll_remote":
            return await self._poll_remote(args)
        elif args.action == "download_remote":
            return await self._download_remote(args)
        
        return ToolResult(
            data=None,
            success=False,
            error=f"Unknown action: {args.action}"
        )
    
    # ── Local (Mock) Operations ──────────────────────────────────
    
    def _submit_local(self, args: JobToolInput) -> ToolResult:
        if not args.script_path:
            return ToolResult(data=None, success=False, error="script_path is required for submit")
        
        script = Path(args.script_path)
        if not script.exists():
            return ToolResult(data=None, success=False, error=f"Script not found: {script}")
        
        output = JobToolOutput(
            job_id=f"mock_{hash(script.name) % 100000:05d}",
            status="submitted",
            output_path=str(script.parent / f"{script.stem}.out"),
            message="Local mock submission. Set HPC config for real remote submission.",
        )
        return ToolResult(data=output.model_dump(), success=True)
    
    def _status_local(self, args: JobToolInput) -> ToolResult:
        if not args.job_id:
            return ToolResult(data=None, success=False, error="job_id is required for status")
        
        output = JobToolOutput(
            job_id=args.job_id,
            status="running",
            runtime="02:34:12",
            message="Local mock status. Use poll_remote for real HPC jobs.",
        )
        return ToolResult(data=output.model_dump(), success=True)
    
    def _cancel_local(self, args: JobToolInput) -> ToolResult:
        if not args.job_id:
            return ToolResult(data=None, success=False, error="job_id is required for cancel")
        
        return ToolResult(
            data={"job_id": args.job_id, "status": "cancelled"},
            success=True,
        )
    
    def _list_local(self, args: JobToolInput) -> ToolResult:
        return ToolResult(
            data={"jobs": [], "note": "Local mock mode. Use submit_remote for real HPC."},
            success=True,
        )
    
    # ── Remote HPC Operations ────────────────────────────────────
    
    def _get_hpc_config(self, args: JobToolInput) -> "HPCConfig":
        """Build HPCConfig from tool args and env vars."""
        from matsci_agent.hpc.client import HPCConfig
        
        host = args.hpc_host or os.environ.get("HPC_HOST")
        username = args.hpc_username or os.environ.get("HPC_USERNAME")
        scheduler = args.hpc_scheduler or os.environ.get("HPC_SCHEDULER", "slurm")
        key_path = args.hpc_key_path or os.environ.get("HPC_KEY_PATH")
        
        if not host:
            raise ValueError("hpc_host not provided. Set HPC_HOST env var or pass hpc_host.")
        if not username:
            raise ValueError("hpc_username not provided. Set HPC_USERNAME env var or pass hpc_username.")
        
        return HPCConfig(
            host=host,
            username=username,
            scheduler=scheduler,
            key_path=key_path,
            remote_work_dir=args.remote_work_dir or "~/matsci_jobs",
        )
    
    async def _submit_remote(self, args: JobToolInput) -> ToolResult:
        """Submit a job to remote HPC via SSH."""
        try:
            cfg = self._get_hpc_config(args)
        except ValueError as e:
            return ToolResult(data=None, success=False, error=str(e))
        
        try:
            from matsci_agent.hpc.client import HPCClient
            
            with HPCClient(cfg) as client:
                # Generate or upload script
                if args.script_path and Path(args.script_path).exists():
                    # Upload local script
                    local_script = Path(args.script_path)
                    job_name = args.job_name or local_script.stem
                    remote_script = f"{cfg.remote_work_dir}/{job_name}.sh"
                    client.upload_file(str(local_script), remote_script)
                    
                    # Read content for submission
                    with open(local_script, "r") as f:
                        script_content = f.read()
                elif args.command:
                    # Generate script from command
                    job_name = args.job_name or "matsci_job"
                    script_content = client.generate_job_script(
                        command=args.command,
                        job_name=job_name,
                        walltime=f"{args.walltime_hours}:00:00",
                        modules=args.modules,
                        env_vars=args.env_vars,
                    )
                else:
                    return ToolResult(
                        data=None,
                        success=False,
                        error="Either script_path or command is required for remote submission"
                    )
                
                job_id = client.submit_job(script_content, job_name=job_name)
                
                output = JobToolOutput(
                    job_id=job_id,
                    status="submitted",
                    message=f"Submitted to {cfg.host} ({cfg.scheduler}). Job ID: {job_id}",
                )
                return ToolResult(data=output.model_dump(), success=True)
        
        except Exception as e:
            return ToolResult(
                data=None,
                success=False,
                error=f"Remote submission failed: {e}"
            )
    
    async def _poll_remote(self, args: JobToolInput) -> ToolResult:
        """Poll status of a remote job."""
        if not args.job_id:
            return ToolResult(data=None, success=False, error="job_id is required for poll_remote")
        
        try:
            cfg = self._get_hpc_config(args)
        except ValueError as e:
            return ToolResult(data=None, success=False, error=str(e))
        
        try:
            from matsci_agent.hpc.client import HPCClient
            
            with HPCClient(cfg) as client:
                status = client.poll_status(args.job_id)
                
                output = JobToolOutput(
                    job_id=status.job_id,
                    status=status.state.lower(),
                    runtime=status.runtime,
                    message=status.message,
                )
                return ToolResult(data=output.model_dump(), success=True)
        
        except Exception as e:
            return ToolResult(
                data=None,
                success=False,
                error=f"Remote poll failed: {e}"
            )
    
    async def _download_remote(self, args: JobToolInput) -> ToolResult:
        """Download files from remote HPC."""
        if not args.remote_path or not args.local_path:
            return ToolResult(
                data=None,
                success=False,
                error="remote_path and local_path are required for download_remote"
            )
        
        try:
            cfg = self._get_hpc_config(args)
        except ValueError as e:
            return ToolResult(data=None, success=False, error=str(e))
        
        try:
            from matsci_agent.hpc.client import HPCClient
            
            with HPCClient(cfg) as client:
                client.download_file(args.remote_path, args.local_path)
                
                return ToolResult(
                    data={"local_path": args.local_path, "remote_path": args.remote_path},
                    success=True,
                )
        
        except Exception as e:
            return ToolResult(
                data=None,
                success=False,
                error=f"Remote download failed: {e}"
            )
