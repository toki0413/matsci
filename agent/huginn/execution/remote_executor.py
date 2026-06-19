"""Remote executor that runs computational tools on HPC/SSH hosts.

Implements the same ``run(cmd, cwd, timeout)`` interface as
``SandboxExecutor``, so simulation tools can be swapped to remote execution
with a single config flag.
"""

from __future__ import annotations

import contextlib
import logging
import shlex
import tarfile
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from huginn.execution.remote_job_store import RemoteJobRecord, RemoteJobStore
from huginn.hpc.client import HPCClient, HPCConfig, JobStatus
from huginn.hpc.resource_selector import ResourceSelector
from huginn.persistence import (
    JSONRemoteJobBackend,
    NullRemoteJobBackend,
    RemoteJobBackend,
)
from huginn.queue import InMemoryTaskBackend, TaskBackend
from huginn.security.sandbox import SandboxResult

logger = logging.getLogger(__name__)

# Kwargs that are meant for the remote scheduler and must not reach subprocess.run.
_REMOTE_SCHEDULER_KWARGS = {
    "queue",
    "walltime",
    "nodes",
    "ntasks_per_node",
    "modules",
    "job_name",
}


class RemoteExecutor:
    """Execute commands on a remote host via SSH/HPC scheduler.

    The executor stages the local working directory to the remote host,
    submits the command as a scheduler job, waits for completion, and then
    downloads stdout/stderr plus result files back to the local directory.

    Jobs are tracked in a ``RemoteJobBackend`` and can optionally be submitted
    through a ``TaskBackend`` so the API event loop is not blocked.
    """

    def __init__(
        self,
        config: HPCConfig,
        job_store: RemoteJobStore | None = None,
        task_backend: TaskBackend | None = None,
        remote_job_backend: RemoteJobBackend | None = None,
    ):
        self.config = config
        self._client = HPCClient(config)
        self._job_store = job_store
        self._jobs: dict[str, RemoteJobRecord] = {}

        if remote_job_backend is None:
            if job_store is not None:
                remote_job_backend = JSONRemoteJobBackend(
                    path=job_store.path,
                    workspace=job_store.path.parent,
                )
            else:
                remote_job_backend = NullRemoteJobBackend()
        self._remote_job_backend = remote_job_backend

        for record in self._remote_job_backend.load():
            self._jobs[record.local_id] = record

        self._task_backend = task_backend or InMemoryTaskBackend()
        self._task_backend.register_task(
            "huginn.remote.execute", self._execute_remote_job
        )

    def submit(
        self,
        cmd: list[str],
        cwd: str | Path | None = None,
        timeout: float | None = None,
        capture_output: bool = True,
        text: bool = True,
        env: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> str:
        """Submit ``cmd`` to the remote task queue and return the local job ID.

        Callers can later poll or wait with ``wait_for(local_job_id)``. This
        method does not block on scheduler completion.
        """
        local_cwd = str(Path(cwd).resolve() if cwd else Path.cwd())
        local_job_id = kwargs.get("job_name") or uuid.uuid4().hex[:12]
        remote_job_dir = f"{self.config.remote_work_dir}/{local_job_id}"
        safe_job_name = f"huginn_{local_job_id}"

        selection = ResourceSelector(self.config).select(
            queue=kwargs.get("queue"),
            gpu=kwargs.get("gpu"),
            profile=kwargs.get("profile"),
            gpus_per_node=kwargs.get("gpus_per_node"),
        )

        self._jobs[local_job_id] = RemoteJobRecord(
            local_id=local_job_id,
            scheduler_id="",
            command=cmd,
            cwd=local_cwd,
            queue=selection.queue,
            status="QUEUED",
            submitted_at=time.time(),
        )

        self._task_backend.send_task(
            "huginn.remote.execute",
            args=(
                local_job_id,
                cmd,
                local_cwd,
                remote_job_dir,
                safe_job_name,
                selection.queue,
                selection.gpus_per_node,
            ),
            kwargs={
                "task_kwargs": kwargs,
                "env": env,
                "timeout": timeout,
            },
            task_id=local_job_id,
        )
        return local_job_id

    def wait_for(
        self,
        local_job_id: str,
        timeout: float | None = None,
    ) -> SandboxResult:
        """Block until the queued remote job finishes and return its result."""
        task_result = self._task_backend.wait_for(
            local_job_id, timeout=timeout, poll_interval=5.0
        )

        if task_result.status == "SUCCESS":
            return task_result.result

        record = self._jobs.get(local_job_id)
        command = record.command if record is not None else []
        error = task_result.error or "Remote execution failed"
        if record is not None:
            record.status = "FAILED"
            record.message = error
            self._persist_record(record)

        return SandboxResult(
            success=False,
            returncode=-1,
            stdout="",
            stderr=error,
            command=command,
            dry_run=False,
        )

    def run(
        self,
        cmd: list[str],
        cwd: str | Path | None = None,
        timeout: float | None = None,
        capture_output: bool = True,
        text: bool = True,
        env: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> SandboxResult:
        """Run ``cmd`` on the remote host inside a scheduler job.

        ``cwd`` is staged to the remote host before execution and result files
        are downloaded back afterwards. This method submits the work to the
        configured ``TaskBackend`` and waits for completion, keeping the call
        synchronous for compatibility with the ``SandboxExecutor`` interface.

        Extra keyword arguments are interpreted as scheduler/resource hints:
        ``queue``, ``walltime``, ``nodes``, ``ntasks_per_node``, ``modules``,
        ``job_name``.
        """
        local_job_id = self.submit(
            cmd,
            cwd=cwd,
            timeout=timeout,
            capture_output=capture_output,
            text=text,
            env=env,
            **kwargs,
        )
        return self.wait_for(local_job_id, timeout=timeout)

    def _execute_remote_job(
        self,
        local_job_id: str,
        cmd: list[str],
        local_cwd_str: str,
        remote_job_dir: str,
        safe_job_name: str,
        queue: str | None,
        gpus_per_node: int,
        task_kwargs: dict[str, Any],
        env: dict[str, str] | None,
        timeout: float | None,
    ) -> SandboxResult:
        """Body of the remote job that is dispatched by the task backend."""
        local_cwd = Path(local_cwd_str)

        def _execute() -> SandboxResult:
            self._client.connect()
            self._stage_to_remote(local_cwd, remote_job_dir)

            command = f"cd {shlex.quote(remote_job_dir)} && {' '.join(shlex.quote(c) for c in cmd)}"
            if env:
                exports = " ".join(
                    f"export {shlex.quote(k)}={shlex.quote(v)}" for k, v in env.items()
                )
                command = f"{exports} && {command}"

            script = self._client.generate_job_script(
                command=command,
                job_name=safe_job_name,
                queue=queue,
                walltime=task_kwargs.get("walltime"),
                nodes=task_kwargs.get("nodes"),
                ntasks_per_node=task_kwargs.get("ntasks_per_node"),
                modules=task_kwargs.get("modules"),
                env_vars=None,
                gpus_per_node=gpus_per_node,
            )
            scheduler_job_id = self._client.submit_job(script, job_name=safe_job_name)

            record = RemoteJobRecord(
                local_id=local_job_id,
                scheduler_id=scheduler_job_id,
                command=cmd,
                cwd=str(local_cwd),
                queue=queue,
                status="PENDING",
                submitted_at=time.time(),
            )
            self._jobs[local_job_id] = record
            self._persist_record(record)

            poll_timeout = int(timeout) if timeout else 86400
            status = self._client.wait_for_job(
                scheduler_job_id,
                poll_interval=10,
                timeout=poll_timeout,
            )
            self._update_record(record, status)

            self._fetch_from_remote(remote_job_dir, local_cwd, scheduler_job_id)
            stdout, stderr = self._read_job_outputs(local_cwd, scheduler_job_id)

            return SandboxResult(
                success=status.exit_code == 0,
                returncode=status.exit_code if status.exit_code is not None else -1,
                stdout=stdout,
                stderr=stderr,
                command=cmd,
                dry_run=False,
            )

        try:
            return self._with_retry(_execute)
        except Exception as exc:
            logger.exception("Remote execution failed")
            return SandboxResult(
                success=False,
                returncode=-1,
                stdout="",
                stderr=f"Remote execution failed: {exc}",
                command=cmd,
                dry_run=False,
            )
        finally:
            with contextlib.suppress(Exception):
                self._client.disconnect()

    def _with_retry(
        self,
        func: Any,
        max_retries: int | None = None,
        backoff: float | None = None,
    ) -> SandboxResult:
        """Retry transient SSH/scheduler failures."""
        max_retries = (
            max_retries if max_retries is not None else self.config.max_retries
        )
        backoff = backoff if backoff is not None else self.config.retry_backoff
        last_exc: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                return func()
            except Exception as exc:
                last_exc = exc
                msg = str(exc).lower()
                transient = any(
                    tag in msg
                    for tag in (
                        "timeout",
                        "timed out",
                        "connection",
                        "eof occurred",
                        "temporarily unavailable",
                        "ssh",
                    )
                )
                if not transient or attempt == max_retries:
                    raise
                wait = backoff * (2 ** (attempt - 1))
                logger.warning(
                    "Remote execution transient error (attempt %d/%d): %s. Retrying in %.1fs",
                    attempt,
                    max_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)
        raise last_exc  # pragma: no cover

    def _update_record(self, record: RemoteJobRecord, status: JobStatus) -> None:
        record.status = status.state
        record.exit_code = status.exit_code
        record.message = status.message
        if status.state in ("COMPLETED", "FAILED", "CANCELLED"):
            record.completed_at = time.time()
        self._persist_record(record)

    def _persist_record(self, record: RemoteJobRecord) -> None:
        self._remote_job_backend.add_or_update(record)

    def list_jobs(self) -> list[RemoteJobRecord]:
        """Return all tracked remote jobs, newest first."""
        return sorted(self._jobs.values(), key=lambda j: j.submitted_at, reverse=True)

    def get_job(self, local_id: str) -> RemoteJobRecord | None:
        """Return a tracked job by its local ID."""
        return self._jobs.get(local_id)

    def refresh_job(self, local_id: str) -> RemoteJobRecord | None:
        """Poll the scheduler for the latest status of a tracked job."""
        record = self._jobs.get(local_id)
        if record is None:
            return None
        try:
            self._client.connect()
            status = self._client.poll_status(record.scheduler_id)
            self._update_record(record, status)
        finally:
            with contextlib.suppress(Exception):
                self._client.disconnect()
        return record

    def cancel_job(self, local_id: str) -> bool:
        """Cancel a tracked job on the scheduler."""
        record = self._jobs.get(local_id)
        if record is None:
            return False
        try:
            self._client.connect()
            if self.config.scheduler == "slurm":
                self._client._exec(f"scancel {shlex.quote(record.scheduler_id)}")
            elif self.config.scheduler == "pbs":
                self._client._exec(f"qdel {shlex.quote(record.scheduler_id)}")
            record.status = "CANCELLED"
            record.completed_at = time.time()
            self._persist_record(record)
            return True
        except Exception as exc:
            logger.warning("Failed to cancel job %s: %s", local_id, exc)
            return False
        finally:
            with contextlib.suppress(Exception):
                self._client.disconnect()

    def _stage_to_remote(self, local_dir: Path, remote_dir: str) -> None:
        """Tar and upload the local working directory to the remote host."""
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            archive_path = Path(tmp.name)

        try:
            with tarfile.open(archive_path, "w:gz") as tar:
                tar.add(local_dir, arcname=".")

            remote_archive = f"{remote_dir}.tar.gz"
            self._client._exec(f"mkdir -p {shlex.quote(remote_dir)}")
            self._client.upload_file(str(archive_path), remote_archive)
            self._client._exec(
                f"tar -xzf {shlex.quote(remote_archive)} -C {shlex.quote(remote_dir)}"
            )
        finally:
            archive_path.unlink(missing_ok=True)

    def _fetch_from_remote(self, remote_dir: str, local_dir: Path, job_id: str) -> None:
        """Download result files from the remote job directory."""
        remote_archive = f"{remote_dir}.tar.gz"
        self._client._exec(
            f"tar -czf {shlex.quote(remote_archive)} -C {shlex.quote(remote_dir)} ."
        )

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            local_archive = Path(tmp.name)

        try:
            self._client.download_file(remote_archive, str(local_archive))
            with tarfile.open(local_archive, "r:gz") as tar:
                tar.extractall(path=local_dir, filter="fully_trusted")
        finally:
            local_archive.unlink(missing_ok=True)

    def _read_job_outputs(self, local_dir: Path, job_id: str) -> tuple[str, str]:
        """Read scheduler stdout/stderr files downloaded from the remote host."""
        stdout = ""
        stderr = ""

        candidates_out = [
            local_dir / f"slurm-{job_id}.out",
            local_dir / f"pbs-{job_id}.out",
        ]
        candidates_err = [
            local_dir / f"slurm-{job_id}.err",
            local_dir / f"pbs-{job_id}.err",
        ]

        for path in candidates_out:
            if path.exists():
                stdout = path.read_text(encoding="utf-8", errors="ignore")
                break

        for path in candidates_err:
            if path.exists():
                stderr = path.read_text(encoding="utf-8", errors="ignore")
                break

        return stdout, stderr


def build_executor(config: Any) -> Any:
    """Build the right executor from a HuginnConfig.

    Returns a ``RemoteExecutor`` when ``execution_backend == "remote"`` and
    the required HPC host/username are configured. When a container runtime and
    image are configured, wraps the local sandbox in a ``ContainerExecutor``.
    Otherwise returns a local ``SandboxExecutor``.
    """
    from huginn.security import ContainerExecutor, SandboxExecutor

    if getattr(config, "execution_backend", "local") != "remote":
        base = SandboxExecutor()
        runtime = getattr(config, "container_runtime", "none") or "none"
        image = getattr(config, "container_image", None)
        if runtime != "none" and image:
            return ContainerExecutor(runtime=runtime, image=image)
        return base

    host = getattr(config, "hpc_host", None)
    username = getattr(config, "hpc_username", None)
    if not host or not username:
        logger.warning(
            "execution_backend=remote but hpc_host/hpc_username missing; "
            "falling back to local sandbox."
        )
        return SandboxExecutor()

    hpc_config = HPCConfig(
        host=host,
        username=username,
        scheduler=getattr(config, "hpc_scheduler", "slurm"),
        key_path=getattr(config, "hpc_key_path", None),
        password=getattr(config, "hpc_password", None),
        port=getattr(config, "hpc_port", 22),
        remote_work_dir=getattr(config, "remote_work_dir", "~/huginn_jobs"),
        default_queue=getattr(config, "hpc_default_queue", None),
        gpu_queue=getattr(config, "hpc_gpu_queue", None),
        queue_map=getattr(config, "hpc_queue_map", {}),
        default_walltime=getattr(config, "hpc_default_walltime", "24:00:00"),
        default_nodes=getattr(config, "hpc_default_nodes", 1),
        default_ntasks_per_node=getattr(config, "hpc_default_ntasks_per_node", 4),
        default_gpus_per_node=getattr(config, "hpc_default_gpus_per_node", 0),
        max_retries=getattr(config, "hpc_max_retries", 3),
        retry_backoff=getattr(config, "hpc_retry_backoff", 1.0),
        strict_host_key_checking=getattr(config, "hpc_strict_host_key_checking", True),
    )
    job_store = RemoteJobStore(workspace=config.workspace)
    return RemoteExecutor(hpc_config, job_store=job_store)
