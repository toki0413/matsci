"""HPC client for remote job submission via SSH.

Supports SLURM (sbatch) and PBS (qsub) schedulers.
"""

from __future__ import annotations

import contextlib
import logging
import re
import shlex
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)


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
    if (
        not path
        or ";" in path
        or "|" in path
        or "&" in path
        or "`" in path
        or "$" in path
    ):
        raise ValueError(f"Path contains forbidden characters: {path!r}")


# Characters/patterns that enable shell injection in job scripts.
# Newlines let an attacker append arbitrary script lines; backticks and
# $() enable command substitution; semicolons chain commands.
_COMMAND_FORBIDDEN = ("\n", "\r", "\0", "`", "$(", "${")

# T-BCSE-13: 常见提权 / 容器逃逸 pattern — 一律拒绝.
#   --privileged / --cap-add / --security-opt / --device / -v(挂载) / --mount
#   是 docker 逃逸; sudo / su / pkexec / runuser 是提权; setuid/setreuid 改 RUID.
_PRIVILEGE_PATTERNS: tuple[str, ...] = (
    "--privileged",
    "--cap-add",
    "--security-opt",
    "--device=",
    "--mount",
    "--user root",
    "--user=root",
    "-u root",
    "sudo",
    "pkexec",
    "runuser",
    "su -",
    "setuid",
    "setreuid",
    "seteuid",
    "chmod 4777",
)


def _validate_command(command: str) -> None:
    """Reject commands containing shell-injection vectors.

    Allows normal program calls with arguments and redirection (>, <, >>)
    but blocks newline injection, command substitution, and chaining.
    Also rejects privilege-escalation / container-escape patterns (T-BCSE-13).
    """
    if not command or not command.strip():
        raise ValueError("HPC command must not be empty")
    if len(command) > 4096:
        raise ValueError("HPC command too long (max 4096 chars)")
    for token in _COMMAND_FORBIDDEN:
        if token in command:
            raise ValueError(f"HPC command contains forbidden sequence {token!r}")
    # Block semicolon chaining — use separate job steps instead
    if ";" in command:
        raise ValueError("HPC command must not contain ';' (use separate steps)")
    # T-BCSE-13: 提权 / 容器逃逸 pattern 拦截 (deny 优先, 不依赖 shell 注入检测)
    low = command.lower()
    for pat in _PRIVILEGE_PATTERNS:
        if pat in low:
            raise ValueError(
                f"HPC command contains forbidden privilege pattern {pat!r}"
            )


def _validate_module_name(module: str) -> None:
    """Module load names should be plain identifiers (e.g. cuda/12.1)."""
    if not module or not re.match(r"^[a-zA-Z0-9_./+-]+$", module):
        raise ValueError(f"Invalid module name: {module!r}")


def _validate_env_var(key: str, value: str) -> None:
    """Env var keys must be valid identifiers; values must be shell-inert.

    Env values are rendered unquoted as ``export KEY=<value>`` in the job
    script, so a value carrying ``$(...)`` / backticks / ``;`` would execute
    on the compute node — the same shell-injection surface ``_validate_command``
    guards for the job command. Reject the same sequences here so the two
    entry points stay consistent.
    """
    if not key or not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", key):
        raise ValueError(f"Invalid env var name: {key!r}")
    for seq in (*_COMMAND_FORBIDDEN, ";"):
        if seq in value:
            raise ValueError(
                f"Env var {key!r} value contains forbidden sequence {seq!r}"
            )


# 优先级到数值的映射; SLURM priority 范围 0-10000, 这里取经验值
# 让 urgent 作业明显比 normal 先跑, 但不抢光集群
_PRIORITY_MAP = {"low": 100, "normal": 500, "high": 1000, "urgent": 2000}


def _priority_to_int(priority: str) -> int | None:
    """把 low/normal/high/urgent 映射成调度器能识别的数值。"""
    return _PRIORITY_MAP.get(priority)


def _validate_dependency_type(dtype: str) -> None:
    """依赖类型必须是 SLURM/PBS 认可的几个值。"""
    if dtype not in ("afterok", "afterany", "afternotok", "after"):
        raise ValueError(f"Unsupported dependency type: {dtype!r}")


def _validate_array_spec(spec: str) -> None:
    """数组规格只允许数字、横杠、逗号和百分号 (限并发)。

    合法例子: "1-10" "1-10%2" "1,3,5" "0-99"
    """
    if not re.match(r"^[0-9][0-9,\-%]*$", spec):
        raise ValueError(f"Invalid array spec: {spec!r}")


def _validate_job_inputs(
    command: str,
    job_name: str,
    modules: list[str] | None,
    env_vars: dict[str, str] | None,
) -> str:
    """校验所有用户可控输入, 返回清洗后的作业名."""
    # Validate all user-controlled inputs before building the script
    _validate_command(command)
    job_name = _sanitize_job_name(job_name)
    if modules:
        for mod in modules:
            _validate_module_name(mod)
    if env_vars:
        for k, v in env_vars.items():
            _validate_env_var(k, str(v))
    return job_name


def _slurm_array_lines(array_spec: str | None) -> list[str]:
    """数组作业: 一条脚本提交一批子任务, 用 %j 区分输出文件."""
    if array_spec:
        _validate_array_spec(array_spec)
        return [
            f"#SBATCH --array={array_spec}",
            "#SBATCH --output=slurm-%A_%a.out",
            "#SBATCH --error=slurm-%A_%a.err",
        ]
    return ["#SBATCH --output=slurm-%j.out", "#SBATCH --error=slurm-%j.err"]


def _slurm_priority_lines(priority: str) -> list[str]:
    """优先级: 映射成数值, urgent 抢前面."""
    prio = _priority_to_int(priority)
    if prio is None or priority == "normal":
        return []
    return [f"#SBATCH --priority={prio}"]


def _slurm_dependency_lines(
    depends_on: list[str] | None, dependency_type: str
) -> list[str]:
    """依赖: 等前置作业跑完再启动, 链式提交靠这个."""
    if not depends_on:
        return []
    _validate_dependency_type(dependency_type)
    dep_str = ":".join(str(j) for j in depends_on)
    return [f"#SBATCH --dependency={dependency_type}:{dep_str}"]


def _pbs_array_lines(array_spec: str | None) -> list[str]:
    """PBS 数组: 不同版本用 -t (Torque) 或 -J (PBS Pro), 这里用兼容性最好的 -t."""
    if array_spec:
        _validate_array_spec(array_spec)
        return [
            f"#PBS -t {array_spec}",
            "#PBS -o pbs-${PBS_ARRAYID}.out",
            "#PBS -e pbs-${PBS_ARRAYID}.err",
        ]
    return ["#PBS -o pbs-$PBS_JOBID.out", "#PBS -e pbs-$PBS_JOBID.err"]


def _pbs_priority_lines(priority: str) -> list[str]:
    """优先级: PBS 用 -p 取整数, 范围 -1024..1023."""
    prio = _priority_to_int(priority)
    if prio is None or priority == "normal":
        return []
    # 映射到 PBS 范围内, urgent 最高
    pbs_prio = {100: -512, 500: 0, 1000: 512, 2000: 1023}
    return [f"#PBS -p {pbs_prio.get(prio, 0)}"]


def _pbs_dependency_lines(
    depends_on: list[str] | None, dependency_type: str
) -> list[str]:
    """依赖: PBS 用 -W depend=afterok:JOBID."""
    if not depends_on:
        return []
    _validate_dependency_type(dependency_type)
    dep_str = ":".join(str(j) for j in depends_on)
    return [f"#PBS -W depend={dependency_type}:{dep_str}"]


def _module_env_lines(
    modules: list[str] | None, env_vars: dict[str, str] | None
) -> list[str]:
    """module load 与 export 行; 每段末尾补一个空行 (两调度器格式一致)."""
    lines: list[str] = []
    if modules:
        for mod in modules:
            lines.append(f"module load {mod}")
        lines.append("")
    if env_vars:
        for key, value in env_vars.items():
            lines.append(f"export {key}={value}")
        lines.append("")
    return lines


@dataclass
class HPCConfig:
    """Configuration for HPC connection and resource selection."""

    host: str
    username: str
    scheduler: Literal["slurm", "pbs"] = "slurm"
    key_path: str | None = None
    password: str | None = None
    port: int = 22
    remote_work_dir: str = "~/huginn_jobs"
    default_queue: str | None = None
    gpu_queue: str | None = None
    queue_map: dict[str, str] = field(default_factory=dict)
    default_walltime: str = "24:00:00"
    default_nodes: int = 1
    default_ntasks_per_node: int = 4
    default_gpus_per_node: int = 0
    max_retries: int = 3
    retry_backoff: float = 1.0
    strict_host_key_checking: bool = True
    known_hosts_path: str | None = None


@dataclass
class JobStatus:
    """Status of a remote HPC job."""

    job_id: str
    state: Literal["PENDING", "RUNNING", "COMPLETED", "FAILED", "CANCELLED", "UNKNOWN"]
    exit_code: int | None = None
    queue: str | None = None
    runtime: str | None = None
    message: str | None = None


@dataclass
class _JobScriptSpec:
    """作业脚本的全部输入参数 (打包后便于拆分渲染 helper)."""

    command: str
    job_name: str
    walltime: str | None
    nodes: int | None
    ntasks_per_node: int | None
    queue: str | None
    modules: list[str] | None
    env_vars: dict[str, str] | None
    gpus_per_node: int | None
    priority: str
    depends_on: list[str] | None
    dependency_type: str
    array_spec: str | None


def _parse_sacct_line(job_id: str, line: str) -> JobStatus | None:
    """解析一行 sacct 输出 "12345|COMPLETED|0:0|normal|01:23:45"."""
    parts = line.split("|")
    if len(parts) < 5:
        return None
    exit_str = parts[2]
    exit_code = int(exit_str.split(":")[0]) if ":" in exit_str else int(exit_str)
    return JobStatus(
        job_id=job_id,
        state=parts[1].upper(),
        exit_code=exit_code,
        queue=parts[3] if parts[3] else None,
        runtime=parts[4] if parts[4] else None,
    )


def _software_result(status: str, method: str, message: str) -> dict[str, str]:
    """构造 install_software 的返回结构 {status, method, message}."""
    return {"status": status, "method": method, "message": message}


class HPCClient:
    """SSH-based HPC client for job submission and monitoring."""

    def __init__(self, config: HPCConfig, pool=None):
        """Create an HPC client.

        ``pool`` is an optional ``SSHConnectionPool``. When provided,
        ``connect()``/``disconnect()`` route through the pool instead of
        opening a fresh connection every time — so ``_exec``/``poll_status``
        reuse one SSH connection across calls (T1: 连接池复用).
        """
        self.config = config
        self._ssh = None
        self._sftp = None
        self._pool = pool
        self._pool_owner = None  # 池中借出的 client (borrowed 时非 None)
        self._borrowed = False

    def connect(self, timeout: int = 10) -> None:
        """Establish SSH connection to the HPC host.

        When ``strict_host_key_checking`` is enabled (default), the host key
        must be present in the known_hosts file; unknown hosts are rejected.

        If a connection pool was injected, borrow a connected client from it
        instead of opening a fresh SSH connection (T1: 连接池复用).
        """
        if self._pool is not None:
            owner = self._pool.get_client(self.config)
            self._pool_owner = owner
            self._ssh = owner._ssh
            self._sftp = owner._sftp
            self._borrowed = True
            return

        import paramiko

        self._ssh = paramiko.SSHClient()

        if self.config.strict_host_key_checking:
            self._ssh.set_missing_host_key_policy(paramiko.RejectPolicy())
            self._load_known_hosts()
        else:
            # strict_host_key_checking=False 时降级到 AutoAddPolicy,
            # 方便首次连接开发环境. 默认走 RejectPolicy, 见上方分支.
            self._ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # nosec B507

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

    def _load_known_hosts(self) -> None:
        """Load host keys from configured or default known_hosts file."""
        import paramiko

        known_hosts = self.config.known_hosts_path
        if known_hosts:
            self._ssh.load_host_keys(known_hosts)
        else:
            with contextlib.suppress(OSError):
                self._ssh.load_system_host_keys()

        if not self._ssh.get_host_keys().keys():
            raise paramiko.SSHException(
                "Strict host key checking is enabled but no known_hosts entries were loaded."
            )

    def disconnect(self) -> None:
        """Close (or release) the SSH connection.

        If the connection was borrowed from a pool, release it back so the
        pool can reuse it on the next call instead of tearing it down.
        """
        if self._borrowed and self._pool is not None and self._pool_owner is not None:
            self._pool.release_client(self.config, self._pool_owner)
            self._pool_owner = None
            self._ssh = None
            self._sftp = None
            self._borrowed = False
            return
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

        T2: 连接层瞬时故障 (SSHException / socket.timeout) 自动重试一次 —
        丢弃借出的坏连接让池清理, 重连后重跑命令. 非连接层错误直接抛出.
        """
        try:
            return self._exec_once(command)
        except self._transient_exc_types() as exc:
            logger.warning("SSH exec transient error, retrying once: %s", exc)
            # 丢弃当前坏连接 (池中借出的归还时会被健康检查剔除, 自建的直接关).
            with contextlib.suppress(Exception):
                self.disconnect()
            return self._exec_once(command)

    @staticmethod
    def _transient_exc_types() -> tuple[type[BaseException], ...]:
        """连接层瞬时异常类型集合 (paramiko 惰性取, 未安装时仅 socket/OSError)."""
        types: list[type[BaseException]] = [socket.timeout, OSError]
        try:
            import paramiko

            types.append(paramiko.SSHException)
        except ImportError:
            pass
        return tuple(types)

    def _exec_once(self, command: str | list[str]) -> tuple[str, str, int]:
        self._ensure_connected()
        if isinstance(command, list):
            command = shlex.join(command)
        stdin, stdout, stderr = self._ssh.exec_command(command)
        exit_code = stdout.channel.recv_exit_status()
        return (
            stdout.read().decode("utf-8", errors="ignore").strip(),
            stderr.read().decode("utf-8", errors="ignore").strip(),
            exit_code,
        )

    # ── Job Script Generation ─────────────────────────────────────

    def generate_job_script(
        self,
        command: str,
        job_name: str = "huginn_job",
        walltime: str | None = None,
        nodes: int | None = None,
        ntasks_per_node: int | None = None,
        queue: str | None = None,
        modules: list[str] | None = None,
        env_vars: dict[str, str] | None = None,
        gpus_per_node: int | None = None,
        priority: str = "normal",
        depends_on: list[str] | None = None,
        dependency_type: str = "afterok",
        array_spec: str | None = None,
    ) -> str:
        """Generate a job script for the configured scheduler.

        新增参数:
        - priority: low / normal / high / urgent, 映射成调度器优先级
        - depends_on: 前置作业 ID 列表, 全部满足后才启动本作业
        - dependency_type: afterok (默认) / afterany / afternotok
        - array_spec: 数组作业规格 (如 "1-10"), 提交一批同类作业
        """
        job_name = _validate_job_inputs(command, job_name, modules, env_vars)
        spec = _JobScriptSpec(
            command=command,
            job_name=job_name,
            walltime=walltime,
            nodes=nodes,
            ntasks_per_node=ntasks_per_node,
            queue=queue,
            modules=modules,
            env_vars=env_vars,
            gpus_per_node=gpus_per_node,
            priority=priority,
            depends_on=depends_on,
            dependency_type=dependency_type,
            array_spec=array_spec,
        )
        if self.config.scheduler == "slurm":
            return self._generate_slurm_script(spec)
        elif self.config.scheduler == "pbs":
            return self._generate_pbs_script(spec)
        else:
            raise ValueError(f"Unsupported scheduler: {self.config.scheduler}")

    def _generate_slurm_script(self, spec: _JobScriptSpec) -> str:
        """渲染 SLURM (sbatch) 作业脚本."""
        lines = ["#!/bin/bash"]
        lines.append(f"#SBATCH --job-name={spec.job_name}")
        lines.append(f"#SBATCH --time={spec.walltime or self.config.default_walltime}")
        lines.append(f"#SBATCH --nodes={spec.nodes or self.config.default_nodes}")
        lines.append(
            f"#SBATCH --ntasks-per-node={spec.ntasks_per_node or self.config.default_ntasks_per_node}"
        )
        lines.extend(self._slurm_resource_lines(spec))
        lines.extend(_slurm_array_lines(spec.array_spec))
        lines.extend(_slurm_priority_lines(spec.priority))
        lines.extend(_slurm_dependency_lines(spec.depends_on, spec.dependency_type))
        lines.append("")
        lines.extend(_module_env_lines(spec.modules, spec.env_vars))
        lines.append(spec.command)
        lines.append("")
        return "\n".join(lines)

    def _slurm_resource_lines(self, spec: _JobScriptSpec) -> list[str]:
        """GPU 与队列资源指令 (gres 在 partition 之前)."""
        lines: list[str] = []
        gpus = self._resolve_gpus(spec.gpus_per_node)
        if gpus > 0:
            lines.append(f"#SBATCH --gres=gpu:{gpus}")
        if spec.queue or self.config.default_queue:
            lines.append(
                f"#SBATCH --partition={spec.queue or self.config.default_queue}"
            )
        return lines

    def _resolve_gpus(self, gpus_per_node: int | None) -> int:
        """未显式指定 GPU 数时回落到配置默认值."""
        if gpus_per_node is not None:
            return gpus_per_node
        return self.config.default_gpus_per_node

    def _generate_pbs_script(self, spec: _JobScriptSpec) -> str:
        """渲染 PBS (qsub) 作业脚本."""
        lines = ["#!/bin/bash"]
        lines.append(f"#PBS -N {spec.job_name}")
        lines.append(
            f"#PBS -l walltime={spec.walltime or self.config.default_walltime}"
        )
        lines.append(f"#PBS -l {self._pbs_node_spec(spec)}")
        if spec.queue or self.config.default_queue:
            lines.append(f"#PBS -q {spec.queue or self.config.default_queue}")
        lines.extend(_pbs_array_lines(spec.array_spec))
        lines.extend(_pbs_priority_lines(spec.priority))
        lines.extend(_pbs_dependency_lines(spec.depends_on, spec.dependency_type))
        lines.append("")
        lines.append("cd $PBS_O_WORKDIR")
        lines.append("")
        lines.extend(_module_env_lines(spec.modules, spec.env_vars))
        lines.append(spec.command)
        lines.append("")
        return "\n".join(lines)

    def _pbs_node_spec(self, spec: _JobScriptSpec) -> str:
        """拼 PBS 的 nodes[:ppn][:ngpus] 资源串."""
        node_spec = (
            f"nodes={spec.nodes or self.config.default_nodes}:"
            f"ppn={spec.ntasks_per_node or self.config.default_ntasks_per_node}"
        )
        gpus = self._resolve_gpus(spec.gpus_per_node)
        if gpus > 0:
            node_spec += f":ngpus={gpus}"
        return node_spec

    # ── Job Submission ────────────────────────────────────────────

    def submit_job(
        self,
        script_content: str,
        job_name: str = "huginn_job",
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
            return self._poll_slurm_status(job_id)
        elif self.config.scheduler == "pbs":
            return self._poll_pbs_status(job_id)
        else:
            raise ValueError(f"Unsupported scheduler: {self.config.scheduler}")

    def _poll_slurm_status(self, job_id: str) -> JobStatus:
        """用 sacct 查询 SLURM 作业状态, 拿不到时回退 squeue."""
        stdout, stderr, rc = self._exec(
            [
                "sacct",
                "-j",
                job_id,
                "--format=JobID,State,ExitCode,Partition,Elapsed",
                "--noheader",
                "-P",
            ]
        )
        if rc != 0 or not stdout:
            # Fallback to squeue for running jobs
            return self._poll_slurm_queue(job_id, stderr)

        # Parse sacct output: "12345|COMPLETED|0:0|normal|01:23:45"
        for line in stdout.strip().split("\n"):
            status = _parse_sacct_line(job_id, line)
            if status is not None:
                return status
        return JobStatus(job_id=job_id, state="UNKNOWN")

    def _poll_slurm_queue(self, job_id: str, stderr: str) -> JobStatus:
        """sacct 不可用时回退到 squeue 查询运行中作业."""
        stdout, _, rc = self._exec(["squeue", "-j", job_id, "-h", "-o", "%T|%i|%r"])
        if rc == 0 and stdout:
            parts = stdout.split("|")
            return JobStatus(
                job_id=job_id,
                state=parts[0].upper() if parts else "UNKNOWN",
                message=parts[2] if len(parts) > 2 else None,
            )
        return JobStatus(job_id=job_id, state="UNKNOWN", message=stderr)

    def _poll_pbs_status(self, job_id: str) -> JobStatus:
        """用 qstat 查询 PBS 作业状态并映射到统一状态."""
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

    # ── Software Installation ─────────────────────────────────────

    # 支持的软件及安装策略: 优先 module, 其次 conda
    # ponytail: 不做源码编译 — HPC 环境编译链太复杂, module/conda 覆盖 90% 场景
    _SUPPORTED_SOFTWARE: dict[str, dict[str, str]] = {
        "gromacs": {
            "module_name": "gromacs",
            "conda_package": "gromacs",
            "check_cmd": "gmx --version",
        },
        "lammps": {
            "module_name": "lammps",
            "conda_package": "lammps",
            "check_cmd": "lmp -h",
        },
        "vasp": {
            "module_name": "vasp",
            "conda_package": "vasp",
            "check_cmd": "vasp_std --version",
        },
        "cp2k": {
            "module_name": "cp2k",
            "conda_package": "cp2k",
            "check_cmd": "cp2k --version",
        },
    }

    def install_software(self, name: str, method: str = "auto") -> dict[str, str]:
        """在远程 HPC 上安装科学计算软件.

        策略 (method):
          auto (默认): module load → conda install → 报失败
          module: 只尝试 module load
          conda: 只尝试 conda install

        返回 {status, method, message}. 不抛异常 — 调用方决定是否继续.
        """
        name = name.strip().lower()
        sw = self._SUPPORTED_SOFTWARE.get(name)
        if sw is None:
            return self._unsupported_software(name)

        self._ensure_connected()
        check = sw["check_cmd"]

        # 0. 先检查是否已装
        _, _, rc = self._exec(check)
        if rc == 0:
            return _software_result("already_installed", "none", f"{name} 已安装")

        methods = ["module", "conda"] if method == "auto" else [method]

        for m in methods:
            if m == "module":
                result = self._install_via_module(sw, check)
                if result is not None:
                    return result
            elif m == "conda":
                return self._install_via_conda(sw, check)

        return _software_result("failed", method, f"{name} 安装失败, 所有策略均未成功")

    def _unsupported_software(self, name: str) -> dict[str, str]:
        """构造 unsupported 结果, 附支持列表."""
        return {
            "status": "unsupported",
            "method": "none",
            "message": f"不支持自动安装 '{name}', 支持列表: {list(self._SUPPORTED_SOFTWARE)}",
        }

    def _install_via_module(
        self, sw: dict[str, str], check: str
    ) -> dict[str, str] | None:
        """尝试 module load; 拿不到可自动加载的版本时返回 None 交给下一策略."""
        mod = sw["module_name"]
        _, _, rc = self._exec(f"module load {mod} && {check}")
        if rc == 0:
            return _software_result("installed", "module", f"module load {mod} 成功")
        avail, _, _ = self._exec(f"module avail {mod} 2>&1")
        if avail.strip():
            return _software_result(
                "manual_required",
                "module",
                f"module {mod} 存在但加载失败, 请手动 module load. 可用版本:\n{avail[:500]}",
            )
        return None

    def _install_via_conda(self, sw: dict[str, str], check: str) -> dict[str, str]:
        """尝试 conda install; 无论成败都返回结果 (不再 fallback)."""
        pkg = sw["conda_package"]
        _, stderr, rc = self._exec(f"conda install -y -c conda-forge {pkg} && {check}")
        if rc == 0:
            return _software_result("installed", "conda", f"conda install {pkg} 成功")
        return _software_result(
            "failed", "conda", f"conda install {pkg} 失败: {stderr[:300]}"
        )

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False
