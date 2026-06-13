"""HPC client for remote job submission via SSH.

Supports SLURM (sbatch) and PBS (qsub) schedulers.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import re
import shlex


def _sanitize_job_name(name: str) -> str:
    """Sanitize job name to prevent shell injection via filenames.

    Only allows alphanumeric, dash, underscore, and dot.
    """
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]", "_", name)
    if not cleaned or cleaned == "_":
        raise ValueError(f"Invalid job name: {name!r}")
    return cleaned[:64]


def _validate_path_component(path: str) -> None:
    """Ensure a path does not contain shell metacharacters."""
    if not path or ";" in path or "|" in path or "&" in path or "`" in path or "$" in path:
        raise ValueError(f"Path contains forbidden characters: {path!r}")


@dataclass
class HPCConfig:
    """Configuration for HPC connection."""
    host: str
    username: str
    scheduler: Literal["slurm", "pbs"] = "slurm"
    key_path: str | None = None
    password: str | None = None
    port: int = 22
    remote_work_dir: str = "~/matsci_jobs"
    default_queue: str | None = None
    default_walltime: str = "24:00:00"
    default_nodes: int = 1
    default_ntasks_per_node: int = 4


@dataclass
class JobStatus:
    """Status of a remote HPC job."""
    job_id: str
    state: Literal["PENDING", "RUNNING", "COMPLETED", "FAILED", "CANCELLED", "UNKNOWN"]
    exit_code: int | None = None
    queue: str | None = None
    runtime: str | None = None
    message: str | None = None


class HPCClient:
    """SSH-based HPC client for job submission and monitoring."""
    
    def __init__(self, config: HPCConfig):
        self.config = config
        self._ssh = None
        self._sftp = None
    
    def connect(self, timeout: int = 10) -> None:
        """Establish SSH connection to the HPC host."""
        import paramiko
        
        self._ssh = paramiko.SSHClient()
        self._ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        connect_kwargs = {
            "hostname": self.config.host,
            "username": self.config.username,
            "port": self.config.port,
            "timeout": timeout,
            "look_for_keys": True,
        }
        
        if self.config.key_path:
            connect_kwargs["key_filename"] = self.config.key_path
        elif self.config.password:
            connect_kwargs["password"] = self.config.password
        
        self._ssh.connect(**connect_kwargs)
        self._sftp = self._ssh.open_sftp()
    
    def disconnect(self) -> None:
        """Close SSH connection."""
        if self._sftp:
            self._sftp.close()
            self._sftp = None
        if self._ssh:
            self._ssh.close()
            self._ssh = None
    
    def _ensure_connected(self) -> None:
        if self._ssh is None or self._ssh.get_transport() is None:
            self.connect()
    
    def _exec(self, command: str | list[str]) -> tuple[str, str, int]:
        """Execute a command on the remote host.

        If a list is provided, each element is shell-quoted automatically
        to prevent injection. Returns (stdout, stderr, exit_code).
        """
        self._ensure_connected()
        if isinstance(command, list):
            command = shlex.join(command)
        stdin, stdout, stderr = self._ssh.exec_command(command)
        exit_code = stdout.channel.recv_exit_status()
        return stdout.read().decode("utf-8", errors="ignore").strip(), \
               stderr.read().decode("utf-8", errors="ignore").strip(), \
               exit_code
    
    # ── Job Script Generation ─────────────────────────────────────
    
    def generate_job_script(
        self,
        command: str,
        job_name: str = "matsci_job",
        walltime: str | None = None,
        nodes: int | None = None,
        ntasks_per_node: int | None = None,
        queue: str | None = None,
        modules: list[str] | None = None,
        env_vars: dict[str, str] | None = None,
    ) -> str:
        """Generate a job script for the configured scheduler."""
        if self.config.scheduler == "slurm":
            return self._generate_slurm_script(
                command, job_name, walltime, nodes, ntasks_per_node, queue, modules, env_vars
            )
        elif self.config.scheduler == "pbs":
            return self._generate_pbs_script(
                command, job_name, walltime, nodes, ntasks_per_node, queue, modules, env_vars
            )
        else:
            raise ValueError(f"Unsupported scheduler: {self.config.scheduler}")
    
    def _generate_slurm_script(
        self,
        command: str,
        job_name: str,
        walltime: str | None,
        nodes: int | None,
        ntasks_per_node: int | None,
        queue: str | None,
        modules: list[str] | None,
        env_vars: dict[str, str] | None,
    ) -> str:
        lines = ["#!/bin/bash"]
        lines.append(f"#SBATCH --job-name={job_name}")
        lines.append(f"#SBATCH --time={walltime or self.config.default_walltime}")
        lines.append(f"#SBATCH --nodes={nodes or self.config.default_nodes}")
        lines.append(f"#SBATCH --ntasks-per-node={ntasks_per_node or self.config.default_ntasks_per_node}")
        
        if queue or self.config.default_queue:
            lines.append(f"#SBATCH --partition={queue or self.config.default_queue}")
        
        lines.append("#SBATCH --output=slurm-%j.out")
        lines.append("#SBATCH --error=slurm-%j.err")
        lines.append("")
        
        if modules:
            for mod in modules:
                lines.append(f"module load {mod}")
            lines.append("")
        
        if env_vars:
            for key, value in env_vars.items():
                lines.append(f"export {key}={value}")
            lines.append("")
        
        lines.append(command)
        lines.append("")
        
        return "\n".join(lines)
    
    def _generate_pbs_script(
        self,
        command: str,
        job_name: str,
        walltime: str | None,
        nodes: int | None,
        ntasks_per_node: int | None,
        queue: str | None,
        modules: list[str] | None,
        env_vars: dict[str, str] | None,
    ) -> str:
        lines = ["#!/bin/bash"]
        lines.append(f"#PBS -N {job_name}")
        lines.append(f"#PBS -l walltime={walltime or self.config.default_walltime}")
        lines.append(f"#PBS -l nodes={nodes or self.config.default_nodes}:ppn={ntasks_per_node or self.config.default_ntasks_per_node}")
        
        if queue or self.config.default_queue:
            lines.append(f"#PBS -q {queue or self.config.default_queue}")
        
        lines.append("#PBS -o pbs-$PBS_JOBID.out")
        lines.append("#PBS -e pbs-$PBS_JOBID.err")
        lines.append("")
        lines.append(f"cd $PBS_O_WORKDIR")
        lines.append("")
        
        if modules:
            for mod in modules:
                lines.append(f"module load {mod}")
            lines.append("")
        
        if env_vars:
            for key, value in env_vars.items():
                lines.append(f"export {key}={value}")
            lines.append("")
        
        lines.append(command)
        lines.append("")
        
        return "\n".join(lines)
    
    # ── Job Submission ────────────────────────────────────────────
    
    def submit_job(
        self,
        script_content: str,
        job_name: str = "matsci_job",
    ) -> str:
        """Submit a job script and return the job ID."""
        self._ensure_connected()

        # Sanitize inputs
        safe_job_name = _sanitize_job_name(job_name)
        _validate_path_component(self.config.remote_work_dir)

        # Create remote work directory
        self._exec(["mkdir", "-p", self.config.remote_work_dir])

        # Write script to remote
        remote_script = f"{self.config.remote_work_dir}/{safe_job_name}.sh"

        # Use sftp to write file
        with self._sftp.file(remote_script, "w") as f:
            f.write(script_content)

        self._exec(["chmod", "+x", remote_script])

        # Submit
        if self.config.scheduler == "slurm":
            stdout, stderr, rc = self._exec(
                ["cd", self.config.remote_work_dir, "&&", "sbatch", remote_script]
            )
            if rc != 0:
                raise RuntimeError(f"sbatch failed: {stderr}")
            # Parse job ID: "Submitted batch job 12345"
            import re
            match = re.search(r"Submitted batch job (\d+)", stdout)
            if match:
                return match.group(1)
            raise RuntimeError(f"Could not parse job ID from: {stdout}")

        elif self.config.scheduler == "pbs":
            stdout, stderr, rc = self._exec(
                ["cd", self.config.remote_work_dir, "&&", "qsub", remote_script]
            )
            if rc != 0:
                raise RuntimeError(f"qsub failed: {stderr}")
            # PBS returns just the job ID
            return stdout.strip().split(".")[0]

        else:
            raise ValueError(f"Unsupported scheduler: {self.config.scheduler}")
    
    # ── Job Polling ───────────────────────────────────────────────
    
    def poll_status(self, job_id: str) -> JobStatus:
        """Check the status of a submitted job."""
        self._ensure_connected()
        
        if self.config.scheduler == "slurm":
            stdout, stderr, rc = self._exec(
                ["sacct", "-j", job_id, "--format=JobID,State,ExitCode,Partition,Elapsed", "--noheader", "-P"]
            )
            if rc != 0 or not stdout:
                # Fallback to squeue for running jobs
                stdout2, stderr2, rc2 = self._exec(
                    ["squeue", "-j", job_id, "-h", "-o", "%T|%i|%r"]
                )
                if rc2 == 0 and stdout2:
                    parts = stdout2.split("|")
                    return JobStatus(
                        job_id=job_id,
                        state=parts[0].upper() if parts else "UNKNOWN",
                        message=parts[2] if len(parts) > 2 else None,
                    )
                return JobStatus(job_id=job_id, state="UNKNOWN", message=stderr)
            
            # Parse sacct output: "12345|COMPLETED|0:0|normal|01:23:45"
            lines = stdout.strip().split("\n")
            for line in lines:
                parts = line.split("|")
                if len(parts) >= 5:
                    state = parts[1].upper()
                    exit_str = parts[2]
                    exit_code = int(exit_str.split(":")[0]) if ":" in exit_str else int(exit_str)
                    return JobStatus(
                        job_id=job_id,
                        state=state,
                        exit_code=exit_code,
                        queue=parts[3] if parts[3] else None,
                        runtime=parts[4] if parts[4] else None,
                    )
            return JobStatus(job_id=job_id, state="UNKNOWN")
        
        elif self.config.scheduler == "pbs":
            stdout, stderr, rc = self._exec(["qstat", "-f", job_id])
            if rc != 0:
                # Job may have finished
                stdout2, stderr2, rc2 = self._exec(["qstat", "-x", "-f", job_id])
                if rc2 != 0:
                    return JobStatus(job_id=job_id, state="UNKNOWN", message=stderr)
                stdout = stdout2
            
            # Parse PBS qstat output
            import re
            state_match = re.search(r"job_state\s*=\s*(\w+)", stdout)
            state = state_match.group(1).upper() if state_match else "UNKNOWN"
            
            exit_match = re.search(r"exit_status\s*=\s*(\d+)", stdout)
            exit_code = int(exit_match.group(1)) if exit_match else None
            
            # Map PBS states
            pbs_state_map = {
                "Q": "PENDING",
                "R": "RUNNING",
                "C": "COMPLETED",
                "E": "RUNNING",  # Exiting
                "H": "PENDING",  # Held
            }
            mapped_state = pbs_state_map.get(state, state)
            
            return JobStatus(
                job_id=job_id,
                state=mapped_state,
                exit_code=exit_code,
            )
        
        else:
            raise ValueError(f"Unsupported scheduler: {self.config.scheduler}")
    
    def wait_for_job(
        self,
        job_id: str,
        poll_interval: int = 30,
        timeout: int = 86400,
    ) -> JobStatus:
        """Poll a job until it completes or times out."""
        start = time.time()
        while time.time() - start < timeout:
            status = self.poll_status(job_id)
            if status.state in ("COMPLETED", "FAILED", "CANCELLED"):
                return status
            time.sleep(poll_interval)
        
        return JobStatus(job_id=job_id, state="UNKNOWN", message="Polling timeout")
    
    # ── File Operations ───────────────────────────────────────────
    
    def download_file(self, remote_path: str, local_path: str) -> None:
        """Download a file from the remote host."""
        self._ensure_connected()
        self._sftp.get(remote_path, local_path)
    
    def upload_file(self, local_path: str, remote_path: str) -> None:
        """Upload a file to the remote host."""
        self._ensure_connected()
        # Ensure parent directory exists
        remote_dir = str(Path(remote_path).parent).replace("\\", "/")
        _validate_path_component(remote_dir)
        self._exec(["mkdir", "-p", remote_dir])
        self._sftp.put(local_path, remote_path)
    
    def list_remote_files(self, remote_dir: str) -> list[str]:
        """List files in a remote directory."""
        self._ensure_connected()
        _validate_path_component(remote_dir)
        stdout, stderr, rc = self._exec(["ls", "-1", remote_dir])
        if rc != 0:
            return []
        return stdout.strip().split("\n") if stdout.strip() else []
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False
