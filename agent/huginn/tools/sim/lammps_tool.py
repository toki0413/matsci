"""LAMMPS molecular dynamics tool — real execution via subprocess.

Uses the installed lmp.exe for actual MD simulations.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from huginn.security import SandboxExecutor
from huginn.tools.base import HuginnTool, ResearchPhase, ToolProfile
from huginn.types import HandleType, ToolContext, ToolResult, ValidationResult
from huginn.validation.handle_validator import HandleValidator

try:
    import huginn_ext

    _HAS_HUGINN_EXT = True
except ImportError:
    huginn_ext = None
    _HAS_HUGINN_EXT = False

logger = logging.getLogger(__name__)

# submit_async 实际可以跑的 LAMMPS 计算类型
_LAMMPS_COMPUTE_ACTIONS = ("run", "minimize", "equilibrate")


class LammpsToolInput(BaseModel):
    action: Literal[
        "run",
        "minimize",
        "equilibrate",
        "analyze_trajectory",
        "submit_async",
        "poll_job",
        "wait_job",
        "equilibrium_check",
        "dem_packing",
    ] = Field(...)
    input_script: str = Field(
        default="", description="LAMMPS input script content or file path"
    )
    structure_file: str | None = Field(
        default=None, description="Structure file path (data, xyz, etc.)"
    )
    potentials: list[str] = Field(
        default_factory=list, description="List of potential file paths"
    )
    trajectory_file: str | None = Field(
        default=None, description="Trajectory file to analyze (for analyze_trajectory)"
    )
    output_prefix: str = Field(default="lammps_out")
    num_processes: int = Field(default=1, ge=1)
    working_dir: str | None = Field(default=None)
    fixes: dict[str, str] = Field(
        default_factory=dict,
        description="Auto-applied fixes from diagnosis (e.g., {'timestep': '0.5'})",
    )
    # submit_async 专用: 指定实际跑哪种计算 (run/minimize/equilibrate)
    compute_action: Literal["run", "minimize", "equilibrate"] | None = Field(
        default=None,
        description="For submit_async: which computation to run (run/minimize/equilibrate)",
    )
    # poll_job / wait_job 专用
    job_id: str | None = Field(
        default=None,
        description="For poll_job/wait_job: the job_id returned by submit_async",
    )
    # wait_job 专用: 最长等多久 (秒)
    timeout: float = Field(
        default=3600.0,
        ge=1.0,
        description="For wait_job: max seconds to wait before returning (default 3600)",
    )
    # 计算失败 / 物理审计报错时自动诊断 + 改脚本重试的次数. 0 = 关闭自愈.
    max_auto_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description="On failure or physics-audit error, auto-diagnose + patch script and retry up to N times",
    )
    # equilibrium_check 专用
    log_file_path: str | None = Field(
        default=None,
        description="For equilibrium_check: path to log.lammps (defaults to working_dir/log.lammps)",
    )
    target_temp: float | None = Field(
        default=None,
        description="For equilibrium_check: target temperature in K",
    )
    target_pressure: float | None = Field(
        default=None,
        description="For equilibrium_check: target pressure in bar",
    )
    window: float = Field(
        default=30.0,
        ge=1.0,
        le=100.0,
        description="For equilibrium_check: percentage of trailing steps to use (default 30%)",
    )
    # dem_packing 专用: 离散元颗粒碰撞模拟
    dem_box: list[float] = Field(
        default=[100.0, 100.0, 100.0],
        description="DEM simulation box dimensions in LAMMPS units (Å for real, µm for si)",
    )
    dem_n_particles: int = Field(
        default=1000, ge=1, le=100000,
        description="Number of particles for DEM packing",
    )
    dem_radius: float = Field(
        default=5.0, gt=0,
        description="Particle radius (in LAMMPS length unit)",
    )
    dem_radius_std: float = Field(
        default=0.0, ge=0.0,
        description="Particle radius standard deviation (for polydisperse packing)",
    )
    dem_density: float = Field(
        default=1.0, gt=0,
        description="Particle density (mass/volume in LAMMPS units)",
    )
    dem_youngs: float = Field(
        default=1e6, gt=0,
        description="Young's modulus for Hertzian contact (Pa for si, energy/length³ for real)",
    )
    dem_poisson: float = Field(
        default=0.3, ge=0.0, lt=0.5,
        description="Poisson's ratio for contact model",
    )
    dem_friction: float = Field(
        default=0.5, ge=0.0, le=2.0,
        description="Sliding friction coefficient",
    )
    dem_restitution: float = Field(
        default=0.8, gt=0.0, le=1.0,
        description="Restitution coefficient (bounciness, 1=perfectly elastic)",
    )
    dem_n_steps: int = Field(
        default=100000, ge=100,
        description="Number of DEM simulation steps",
    )
    dem_gravity: float = Field(
        default=0.0,
        description="Gravity acceleration (m/s² for si, or 0 for no gravity)",
    )

    @model_validator(mode="after")
    def _check_action_fields(self) -> "LammpsToolInput":
        """不同 action 需要不同字段, schema 层兜底."""
        if self.action == "submit_async":
            if not self.compute_action:
                raise ValueError(
                    "submit_async requires 'compute_action' (run/minimize/equilibrate)"
                )
            if not self.input_script:
                raise ValueError(
                    "submit_async requires 'input_script' (script content or file path)"
                )
        if self.action in ("poll_job", "wait_job") and not self.job_id:
            raise ValueError(f"action '{self.action}' requires 'job_id'")
        return self


class LammpsToolOutput(BaseModel):
    log_path: str | None = None
    trajectory_path: str | None = None
    thermo_data: dict | None = None
    final_energy: float | None = None
    warnings: list[str] = []


class LammpsTool(HuginnTool):
    """Execute LAMMPS molecular dynamics simulations."""

    name = "lammps_tool"
    category = "sim"
    profile = ToolProfile(
        cost_tier="heavy",
        phases=frozenset({ResearchPhase.EXECUTION}),
        constraint_scope="md",
        light_alternatives=("symbolic_math_tool", "numerical_tool"),
    )
    description = (
        "Run LAMMPS molecular dynamics simulations (minimization, equilibration, production). "
        "Supports async submission via submit_async / poll_job / wait_job for long-running jobs."
    )
    input_schema = LammpsToolInput
    _init_kwargs_map = {"lammps_executable": "lammps_executable"}

    # 异步作业注册表: job_id -> {status, task, result, error, started_at, finished_at}
    # 类级别共享, 跟 VaspTool 一样的模式. 进程重启后状态丢失.
    _async_jobs: dict[str, dict[str, Any]] = {}

    def __init__(
        self,
        lammps_executable: str | None = None,
        sandbox: SandboxExecutor | None = None,
    ):
        super().__init__()
        self.lammps_executable = lammps_executable or self._find_lammps()
        self.sandbox = sandbox or SandboxExecutor()

    def _find_lammps(self) -> str | None:
        """Find LAMMPS executable on the system."""
        import glob

        # Check environment variable
        env_path = os.environ.get("LAMMPS_EXECUTABLE")
        if env_path and Path(env_path).exists():
            return env_path

        # Check PATH
        try:
            import shutil

            exe = shutil.which("lmp")
            if exe:
                return exe
        except Exception:
            logger.debug("suppressed in _find_lammps", exc_info=True)

        # Check common Windows locations (with glob for unicode paths)
        patterns = [
            r"C:\Users\*\OneDrive\*\LAMMPS*\bin\lmp.exe",
            r"C:\Program Files*\LAMMPS*\bin\lmp.exe",
            r"C:\ProgramData\*\LAMMPS*\bin\lmp.exe",
        ]
        for pattern in patterns:
            matches = glob.glob(pattern)
            for m in matches:
                if Path(m).exists():
                    return m

        return None

    def estimate_cost(self, args: LammpsToolInput) -> dict[str, float] | None:
        # poll_job / wait_job 是查询操作, 不消耗计算资源
        if args.action in ("poll_job", "wait_job"):
            return None
        return {"cpu_hours": 2, "walltime_hours": 2}

    async def validate_input(
        self, args: LammpsToolInput, context: ToolContext
    ) -> ValidationResult:
        """Pre-flight: verify required files based on action type.

        poll_job / wait_job 只需要 job_id, 不检查文件. submit_async 跟
        普通计算 action 一样检查 input_script / structure_file / potentials.
        """
        # poll_job / wait_job 不需要文件检查
        if args.action in ("poll_job", "wait_job"):
            return ValidationResult(result=True)

        if args.action == "analyze_trajectory":
            traj = args.trajectory_file or args.input_script
            if not traj:
                return ValidationResult(
                    result=False,
                    message="Trajectory file not specified",
                    error_code=400,
                )
            vr = HandleValidator.validate(HandleType.FILE_PATH, traj, context)
            if not vr.result:
                return ValidationResult(
                    result=False,
                    message=f"Trajectory file not found: {traj}",
                    error_code=404,
                )
        if args.structure_file:
            vr = HandleValidator.validate(
                HandleType.FILE_PATH, args.structure_file, context
            )
            if not vr.result:
                return ValidationResult(
                    result=False,
                    message=f"Structure file not found: {args.structure_file}",
                    error_code=404,
                )
        if args.input_script and Path(args.input_script).suffix in (".lammps", ".in", ".lmp"):
            vr = HandleValidator.validate(
                HandleType.FILE_PATH, args.input_script, context
            )
            if not vr.result:
                return ValidationResult(
                    result=False,
                    message=f"Input script file not found: {args.input_script}",
                    error_code=404,
                )
        for pot in args.potentials:
            vr = HandleValidator.validate(HandleType.FILE_PATH, pot, context)
            if not vr.result:
                return ValidationResult(
                    result=False,
                    message=f"Potential file not found: {pot}",
                    error_code=404,
                )
        return ValidationResult(result=True)

    async def call(self, args: LammpsToolInput, context: ToolContext) -> ToolResult:
        # 异步作业管理动作: 不跑实际计算, 只查/等作业状态
        if args.action == "submit_async":
            return await self._handle_submit_async(args, context)
        if args.action == "poll_job":
            return self._handle_poll_job(args)
        if args.action == "wait_job":
            return await self._handle_wait_job(args)

        # Equilibrium check: analyze thermo data from a log file, no LAMMPS run
        if args.action == "equilibrium_check":
            return self._run_equilibrium_check(args)

        # DEM packing: generate granular input script, then optionally run
        if args.action == "dem_packing":
            return await self._handle_dem_packing(args, context)

        # Handle trajectory analysis without running LAMMPS
        if args.action == "analyze_trajectory":
            traj_file = args.trajectory_file or args.input_script
            if not traj_file or not Path(traj_file).exists():
                return ToolResult(
                    data=None,
                    success=False,
                    error="Trajectory file not specified or not found",
                )
            analysis = self.parse_trajectory(traj_file)
            # 分析也算一次计算, 带 provenance
            try:
                from huginn.provenance import capture

                analysis["provenance"] = capture(
                    "lammps_tool", args.model_dump(), output=dict(analysis)
                ).to_dict()
            except Exception:
                logger.debug("suppressed in estimate_cost", exc_info=True)
            # 轨迹里有 MSD, 提示 agent 用 GP 量化扩散系数的不确定性
            analysis["uq_hint"] = self._uq_hint()
            return ToolResult(
                data=analysis,
                success="error" not in analysis,
                error=analysis.get("error"),
            )

        if not self.lammps_executable:
            from huginn.tools.sim.executable_resolver import resolve_executable, ResolutionRequest

            resolution = resolve_executable("lammps")
            if isinstance(resolution, str):
                self.lammps_executable = resolution
            else:
                return ToolResult(
                    data=None,
                    success=False,
                    error=f"LAMMPS executable not found. {resolution.install_hint}",
                    metadata={"needs_resolution": True, "resolution_request": resolution.to_dict()},
                )

        # Determine working directory
        if args.working_dir:
            work_dir = Path(args.working_dir)
        else:
            work_dir = Path(context.workspace) / f"lammps_{args.output_prefix}"
        work_dir.mkdir(parents=True, exist_ok=True)

        # Write input script
        input_path = work_dir / "input.lammps"

        # Check if input_script is a file path or content
        script_path = Path(args.input_script)
        if script_path.exists():
            script_content = script_path.read_text(encoding="utf-8")
        else:
            script_content = args.input_script

        # Prepend structure read if structure_file provided
        if args.structure_file:
            structure_path = Path(args.structure_file)
            if structure_path.exists():
                # Detect format and prepend read command
                if structure_path.suffix in [".data", ".lmp"]:
                    prefix = f"read_data {structure_path}\n"
                elif structure_path.suffix == ".xyz":
                    prefix = f"read_xyz {structure_path}\n"
                else:
                    prefix = f"read_data {structure_path}\n"

                if (
                    "read_data" not in script_content
                    and "read_xyz" not in script_content
                ):
                    script_content = prefix + script_content

        # Apply auto-fixes from diagnosis to input script
        if args.fixes:
            script_content = self._apply_script_fixes(script_content, args.fixes)

        input_path.write_text(script_content, encoding="utf-8")

        # Copy potential files to working directory
        for pot in args.potentials:
            pot_path = Path(pot)
            if pot_path.exists():
                dest = work_dir / pot_path.name
                if not dest.exists():
                    import shutil

                    shutil.copy2(pot_path, dest)

        # Resolve to absolute paths to avoid relative path issues on Windows
        work_dir_abs = work_dir.resolve()
        input_path_abs = input_path.resolve()
        log_path_abs = (work_dir_abs / "log.lammps").resolve()

        # Build command
        cmd = [
            self.lammps_executable,
            "-in",
            str(input_path_abs),
            "-log",
            str(log_path_abs),
        ]
        if args.num_processes > 1:
            cmd = ["mpiexec", "-n", str(args.num_processes)] + cmd

        # Run LAMMPS — 带 autofix 重试:
        # 硬失败 (returncode!=0) 或物理审计报错 (温度爆炸 / 能量漂移) 都重试
        autoheal_log: list[dict[str, Any]] = []
        max_retries = args.max_auto_retries
        result: Any = None
        thermo_data: dict = {}
        final_energy: float | None = None
        warnings: list[str] = []
        audit_report = None
        # 软失败原因. returncode=0 但物理审计报错属于软失败,
        # 这种情况 result.returncode==0, 不能当成功返回.
        soft_failure_msg: str | None = None

        try:
            for attempt in range(max_retries + 1):
                sb_result = self.sandbox.run(
                    cmd,
                    cwd=str(work_dir_abs),
                    timeout=3600,
                )
                result = sb_result

                # Parse log file for thermo data
                log_path = work_dir / "log.lammps"
                thermo_data, final_energy, warnings = self._parse_log(log_path)

                # 判断这次跑完到底算成功还是失败:
                # - returncode != 0 → 硬失败, stderr 诊断
                # - returncode == 0 但物理审计报错 → 软失败,
                #   LAMMPS 经常 exit=0 但轨迹是垃圾 (温度爆炸 / 能量漂移)
                error: str | None = None
                soft_failure_msg = None
                if result.returncode != 0:
                    error = result.stderr or ""
                else:
                    try:
                        from huginn.execution.physics_auditor import PhysicsAuditor

                        auditor = PhysicsAuditor()
                        audit_report = auditor.audit(
                            "lammps_tool",
                            args.compute_action or args.action,
                            {
                                "thermo_data": thermo_data,
                                "final_energy": final_energy,
                            },
                            args.model_dump(),
                        )
                        if audit_report.has_errors:
                            errs = [
                                f.message
                                for f in audit_report.findings
                                if f.severity == "error"
                            ]
                            error = f"Physics audit found errors: {errs}"
                            soft_failure_msg = error
                    except Exception:
                        logger.debug("审计本身挂了不能阻塞结果", exc_info=True)

                if error is None:
                    break  # 真正成功

                # 失败了 (硬失败或软失败), 看还有没有重试额度 + 能不能自动修
                if attempt < max_retries:
                    fixed = self._try_autofix(input_path_abs, error)
                    if fixed:
                        autoheal_log.append(
                            {
                                "attempt": attempt + 1,
                                "error": error[:300],
                                "fixes_applied": fixed["fixes"],
                                "reasoning": fixed["reasoning"],
                            }
                        )
                        continue
                break  # 没修动或重试耗尽

            # Find trajectory file
            traj_path = None
            for ext in [".lammpstrj", ".dump", ".xyz"]:
                candidates = list(work_dir.glob(f"*{ext}"))
                if candidates:
                    traj_path = str(candidates[0])
                    break

            output = LammpsToolOutput(
                log_path=str(work_dir / "log.lammps"),
                trajectory_path=traj_path,
                thermo_data=thermo_data,
                final_energy=final_energy,
                warnings=warnings,
            )

            # 最终成功判定: returncode==0 且没有遗留软失败 (物理审计报错)
            ok = result.returncode == 0 and soft_failure_msg is None
            error_out = (
                None
                if ok
                else (
                    result.stderr[:500]
                    if result.returncode != 0
                    else soft_failure_msg
                )
            )

            data = output.model_dump()

            # Auto-parse trajectory if available
            if traj_path:
                traj_analysis = self.parse_trajectory(traj_path)
                if "error" not in traj_analysis:
                    data["trajectory_analysis"] = traj_analysis

            # Physics audit — check thermo data for unphysical values.
            # LAMMPS can exit cleanly while the trajectory itself is garbage
            # (e.g. exploded temperatures, runaway pressure). Flag those here.
            # 循环里跑过审计就复用, 没跑过 (比如硬失败) 就补跑一次兜底.
            if audit_report is not None:
                data["physics_audit"] = audit_report.to_dict()
            else:
                try:
                    from huginn.execution.physics_auditor import PhysicsAuditor

                    auditor = PhysicsAuditor()
                    audit_report = auditor.audit(
                        "lammps_tool",
                        args.compute_action or args.action,
                        {
                            "thermo_data": thermo_data,
                            "final_energy": final_energy,
                        },
                        args.model_dump(),
                    )
                    data["physics_audit"] = audit_report.to_dict()
                except Exception:
                    logger.debug("audit is best-effort, never block the result", exc_info=True)

            if autoheal_log:
                data["autoheal_attempts"] = autoheal_log

            # 带上 provenance 快照, 事后能追溯参数/版本/环境
            # 注意: 在加 provenance 字段前先 snapshot 输出, 避免 output_hash 自指
            try:
                from huginn.provenance import capture

                data["provenance"] = capture(
                    "lammps_tool", args.model_dump(), output=dict(data)
                ).to_dict()
            except Exception:
                logger.debug("provenance 失败不能把计算结果带挂", exc_info=True)

            # 提示 agent 用 GP 拟合 MSD-vs-time, 量化扩散系数不确定性
            data["uq_hint"] = self._uq_hint()

            return ToolResult(
                data=data,
                success=ok,
                error=error_out,
            )

        except subprocess.TimeoutExpired:
            return ToolResult(
                data=None, success=False, error="LAMMPS execution timed out (3600s)"
            )
        except Exception as e:
            return ToolResult(
                data=None, success=False, error=f"LAMMPS execution failed: {e}"
            )

    # ------------------------------------------------------------------ async job API

    async def _handle_submit_async(
        self, args: LammpsToolInput, context: ToolContext
    ) -> ToolResult:
        """异步提交 LAMMPS 计算, 立即返回 job_id, 不阻塞.

        内部用 asyncio.create_task 在后台跑实际计算 (run/minimize/equilibrate),
        计算完成后把结果塞进 _async_jobs[job_id]. 进程重启后作业状态会丢,
        长跑作业建议走 job_tool 提交到 HPC.

        Returns:
            ToolResult.data = {"job_id": str, "status": "running", "compute_action": str}
        """
        # 构造同步调用的 args: 用 compute_action 作为 action, 透传其它字段
        sync_args = LammpsToolInput(
            action=args.compute_action,
            input_script=args.input_script,
            structure_file=args.structure_file,
            potentials=args.potentials,
            output_prefix=args.output_prefix,
            num_processes=args.num_processes,
            working_dir=args.working_dir,
            fixes=args.fixes,
        )

        job_id = f"lammps-{uuid.uuid4().hex[:12]}"

        job_entry: dict[str, Any] = {
            "status": "running",
            "task": None,
            "result": None,
            "error": None,
            "compute_action": args.compute_action,
            "working_dir": args.working_dir,
            "started_at": time.time(),
            "finished_at": None,
        }
        LammpsTool._async_jobs[job_id] = job_entry

        async def _run_in_background() -> None:
            """后台跑实际计算, 完成后更新 job_entry."""
            try:
                result = await self.call(sync_args, context)
                job_entry["result"] = (
                    result.data if result.success else None
                )
                job_entry["error"] = result.error
                job_entry["status"] = "done" if result.success else "failed"
            except Exception as exc:
                job_entry["error"] = str(exc)
                job_entry["status"] = "failed"
            finally:
                job_entry["finished_at"] = time.time()

        # create_task 把协程排到当前事件循环, agent chat 期间会并发跑
        try:
            task = asyncio.create_task(_run_in_background())
            job_entry["task"] = task
        except RuntimeError:
            # 没有运行中的事件循环 (比如同步路径调用), 退化为同步执行
            # 这种情况下 "异步" 提交实际是阻塞的, 但至少功能正确
            await _run_in_background()

        return ToolResult(
            data={
                "job_id": job_id,
                "status": "running",
                "compute_action": args.compute_action,
                "working_dir": args.working_dir,
            },
            success=True,
        )

    def _handle_poll_job(self, args: LammpsToolInput) -> ToolResult:
        """查作业状态, 立即返回, 不阻塞.

        Returns:
            ToolResult.data = {
                "job_id": str,
                "status": "running" | "done" | "failed",
                "progress": 0-100,
                "partial_result": ... | None,
                "error": str | None,
                "elapsed": float,
            }
        """
        job_id = args.job_id
        job = LammpsTool._async_jobs.get(job_id)
        if job is None:
            return ToolResult(
                data=None,
                success=False,
                error=f"Unknown job_id: {job_id}",
            )

        elapsed = time.time() - job["started_at"]
        status = job["status"]
        # 进度估算: running 给 50 (LAMMPS 内部进度没法简单跟踪),
        # done/failed 给 100. 真要精确进度得解析 log.lammps 的 thermo step,
        # 这里先做粗略估计, 跟 VaspTool 保持一致.
        progress = 100 if status in ("done", "failed") else 50

        return ToolResult(
            data={
                "job_id": job_id,
                "status": status,
                "progress": progress,
                "partial_result": job["result"] if status == "done" else None,
                "error": job["error"],
                "elapsed": round(elapsed, 2),
                "compute_action": job.get("compute_action"),
            },
            success=True,
        )

    async def _handle_wait_job(self, args: LammpsToolInput) -> ToolResult:
        """阻塞等待作业完成或超时.

        内部用 asyncio.wait_for 等后台 task, 超时返回当前状态 (status=running).
        作业完成返回最终结果, 失败返回错误. 超时不取消 task, 让它继续在后台跑.
        """
        job_id = args.job_id
        timeout = args.timeout
        job = LammpsTool._async_jobs.get(job_id)
        if job is None:
            return ToolResult(
                data=None,
                success=False,
                error=f"Unknown job_id: {job_id}",
            )

        task = job.get("task")
        if task is None:
            # 同步退化路径下没有 task, 直接返回当前状态
            return self._handle_poll_job(args)

        # 已经完成的作业直接返回, 不再 wait
        if job["status"] in ("done", "failed"):
            return self._handle_poll_job(args)

        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
        except asyncio.TimeoutError:
            # 超时: 作业还在跑, 返回当前状态 (不取消 task, 让它继续)
            pass
        except Exception as exc:
            # task 本身挂了 (不是超时), 把错误记下来
            job["status"] = "failed"
            job["error"] = str(exc)
            job["finished_at"] = time.time()

        return self._handle_poll_job(args)

    def _uq_hint(self) -> dict[str, Any]:
        """提示 agent 用 gp_tool 拟合 MSD-vs-time 做不确定性量化."""
        return {
            "tool": "gp_tool",
            "action": "fit",
            "suggestion": (
                "Consider calling gp_tool with action='fit' to fit a Gaussian "
                "Process to the MSD-vs-time trajectory for uncertainty quantification "
                "of the diffusion coefficient."
            ),
            "data_mapping": {
                "X": "time (trajectory timesteps)",
                "y": "msd (mean squared displacement)",
            },
        }

    def _parse_log(self, log_path: Path) -> tuple[dict, float | None, list[str]]:
        """Parse LAMMPS log file for thermodynamic data."""
        if not log_path.exists():
            return {}, None, ["Log file not found"]

        thermo_data = {}
        final_energy = None
        warnings = []

        try:
            content = log_path.read_text(encoding="utf-8", errors="ignore")

            # Identify thermo columns from the header
            # Pattern: Step Temp Press TotEng ...
            header_match = re.search(r"^(Step\s+.*?)$", content, re.MULTILINE)
            columns = []
            if header_match:
                columns = header_match.group(1).split()

            # Extract all thermo data rows
            data_rows = []
            # Match lines that start with an integer step number followed by numeric values
            for line in content.split("\n"):
                parts = line.split()
                if len(parts) >= 2 and parts[0].isdigit():
                    try:
                        # Verify most parts are numeric
                        numeric_count = sum(1 for p in parts if self._is_float(p))
                        if numeric_count >= len(parts) - 1:
                            data_rows.append([self._to_float_or_str(p) for p in parts])
                    except ValueError:
                        logger.debug("suppressed in _parse_log", exc_info=True)

            if data_rows and columns:
                # Transpose: columns[0] is Step, columns[1] is Temp, etc.
                for col_idx, col_name in enumerate(columns):
                    if col_idx < len(data_rows[0]):
                        values = [
                            row[col_idx] for row in data_rows if col_idx < len(row)
                        ]
                        # Try to convert to float
                        float_values = []
                        for v in values:
                            if isinstance(v, float):
                                float_values.append(v)
                            elif isinstance(v, str) and self._is_float(v):
                                float_values.append(float(v))
                        if float_values:
                            thermo_data[col_name.lower()] = float_values

            # Extract final energy
            if "toteng" in thermo_data and thermo_data["toteng"]:
                final_energy = thermo_data["toteng"][-1]
            elif "toteng" not in thermo_data:
                # Fallback: search for TotEng explicitly
                energy_match = re.findall(r"TotEng\s+([-\d.eE]+)", content)
                if energy_match:
                    with contextlib.suppress(ValueError):
                        final_energy = float(energy_match[-1])

            # Check for warnings
            if "WARNING" in content:
                warn_lines = [
                    line.strip() for line in content.split("\n") if "WARNING" in line
                ]
                warnings.extend(warn_lines[:5])

            # Check for errors
            if "ERROR" in content:
                err_lines = [
                    line.strip() for line in content.split("\n") if "ERROR" in line
                ]
                warnings.extend(err_lines[:3])

        except Exception as e:
            warnings.append(f"Failed to parse log: {e}")

        return thermo_data, final_energy, warnings

    def _run_equilibrium_check(self, args: LammpsToolInput) -> ToolResult:
        """Check if an MD run has reached thermal/mechanical equilibrium.

        Parses thermo data from the LAMMPS log, takes the trailing *window*%
        of steps, and checks temperature against *target_temp* (within 5%)
        and drift (linear slope) against a threshold. Returns a recommendation
        if the system hasn't settled yet.
        """
        # resolve which log file to parse
        log_path = Path(args.log_file_path) if args.log_file_path else None
        if log_path is None and args.working_dir:
            log_path = Path(args.working_dir) / "log.lammps"
        if log_path is None or not log_path.exists():
            return ToolResult(
                data=None,
                success=False,
                error="No log file found. Provide log_file_path or working_dir with log.lammps",
            )

        thermo_data, _, _ = self._parse_log(log_path)
        if not thermo_data:
            return ToolResult(
                data={
                    "equilibrated": False,
                    "avg_temp": None,
                    "avg_pressure": None,
                    "temp_drift": None,
                    "pressure_drift": None,
                    "recommendation": "Log file contains no thermo data. Check the log for errors.",
                },
                success=True,
            )

        temps = thermo_data.get("temp", [])
        press = thermo_data.get("press", [])
        steps = thermo_data.get("step", [])

        if not temps:
            return ToolResult(
                data={
                    "equilibrated": False,
                    "avg_temp": None,
                    "avg_pressure": None,
                    "temp_drift": None,
                    "pressure_drift": None,
                    "recommendation": "No temperature data found in log. Check thermo_style.",
                },
                success=True,
            )

        # take the trailing window% of data points
        n_total = len(temps)
        n_tail = max(1, int(n_total * args.window / 100.0))
        tail_temps = temps[-n_tail:]
        tail_press = press[-n_tail:] if press else []
        tail_steps = steps[-n_tail:] if steps else list(range(n_tail))

        avg_temp = sum(tail_temps) / len(tail_temps)
        avg_press = sum(tail_press) / len(tail_press) if tail_press else None

        temp_drift = self._linear_slope(tail_steps, tail_temps)
        pressure_drift = (
            self._linear_slope(tail_steps, tail_press) if tail_press else None
        )

        # temperature within 5% of target?
        temp_ok = True
        if args.target_temp is not None and args.target_temp > 0:
            temp_ok = abs(avg_temp - args.target_temp) / args.target_temp <= 0.05

        # drift threshold: ~1 K per 100 steps is a reasonable cutoff for
        # "still drifting". ponytail: this is heuristic and system-dependent;
        # for production runs, tune based on the specific thermostat/barostat.
        drift_threshold = 0.01
        temp_drift_ok = abs(temp_drift) < drift_threshold

        equilibrated = temp_ok and temp_drift_ok

        # build recommendation
        recommendation = self._build_equilibrium_recommendation(
            equilibrated, avg_temp, args.target_temp, temp_drift,
            avg_press, args.target_pressure, n_tail, n_total,
        )

        return ToolResult(
            data={
                "equilibrated": equilibrated,
                "avg_temp": avg_temp,
                "avg_pressure": avg_press,
                "temp_drift": temp_drift,
                "pressure_drift": pressure_drift,
                "recommendation": recommendation,
                "window_steps": n_tail,
                "total_steps": n_total,
            },
            success=True,
        )

    @staticmethod
    def _build_equilibrium_recommendation(
        equilibrated: bool,
        avg_temp: float,
        target_temp: float | None,
        temp_drift: float,
        avg_press: float | None,
        target_pressure: float | None,
        n_tail: int,
        n_total: int,
    ) -> str:
        if equilibrated:
            return "System has reached equilibrium. Proceed with production run."

        reasons: list[str] = []

        if target_temp is not None and target_temp > 0:
            rel_err = abs(avg_temp - target_temp) / target_temp
            if rel_err > 0.05:
                reasons.append(
                    f"avg temp {avg_temp:.1f} K deviates {rel_err*100:.1f}% from target {target_temp:.1f} K"
                )

        if abs(temp_drift) >= 0.01:
            reasons.append(f"temperature drift {temp_drift:.4f} K/step is too high")

        if target_pressure is not None and avg_press is not None:
            if abs(avg_press - target_pressure) > max(abs(target_pressure) * 0.1, 100.0):
                reasons.append(
                    f"avg pressure {avg_press:.1f} bar is far from target {target_pressure:.1f} bar"
                )

        if not reasons:
            return "System is close to equilibrium. Extend equilibration to confirm stability."

        # suggest extending by 50% more steps or halving the timestep
        extend_by = max(int(n_total * 0.5), 1000)
        rec = "Not equilibrated: " + "; ".join(reasons) + "."
        rec += f" Extend run by ~{extend_by} steps or reduce timestep by half."
        return rec

    @staticmethod
    def _linear_slope(x: list[float], y: list[float]) -> float:
        """Least-squares slope of y vs x. Returns 0 for degenerate input."""
        n = len(y)
        if n < 2:
            return 0.0
        # use list indices as x when x is empty or mismatched length
        if len(x) != n:
            x = list(range(n))
        sx = sum(x)
        sy = sum(y)
        sxy = sum(xi * yi for xi, yi in zip(x, y))
        sxx = sum(xi * xi for xi in x)
        denom = n * sxx - sx * sx
        if denom == 0:
            return 0.0
        return (n * sxy - sx * sy) / denom

    def _apply_script_fixes(self, script: str, fixes: dict[str, str]) -> str:
        """Apply diagnosed fixes to LAMMPS input script.

        Replaces command parameters like 'timestep 1.0' with 'timestep 0.5'.
        """
        lines = script.split("\n")
        modified = []
        applied = set()

        for line in lines:
            stripped = line.strip().lower()
            # Skip comments and blank lines
            if not stripped or stripped.startswith("#"):
                modified.append(line)
                continue

            # Check each fix key
            for key, new_value in fixes.items():
                key_lower = key.lower()
                # Match command at start of line (allow leading whitespace)
                parts = stripped.split()
                if parts and parts[0] == key_lower:
                    # Replace the value part(s)
                    # e.g., 'timestep 1.0' → 'timestep 0.5'
                    # e.g., 'fix nvt all temp 300 300 0.1' → more complex
                    indent = line[: len(line) - len(line.lstrip())]
                    modified_line = f"{indent}{key} {new_value}"
                    modified.append(modified_line)
                    applied.add(key)
                    break
            else:
                modified.append(line)

        # If any fix wasn't applied, append it at the end
        for key, new_value in fixes.items():
            if key.lower() not in applied:
                modified.append(f"{key} {new_value}")

        return "\n".join(modified)

    def _read_script_params(self, input_path: Path) -> dict[str, Any]:
        """读 input.lammps 解析关键参数 (timestep 等), 给 AutoFixLoop 当上下文."""
        params: dict[str, Any] = {}
        try:
            for line in input_path.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                parts = s.split()
                if not parts:
                    continue
                cmd = parts[0].lower()
                # 只抓 AutoFixLoop 关心的几个命令: timestep / neighbor
                if cmd == "timestep" and len(parts) > 1:
                    try:
                        params["timestep"] = float(parts[1])
                    except ValueError:
                        params["timestep"] = parts[1]
                elif cmd == "neighbor" and len(parts) > 1:
                    try:
                        params["neighbor"] = float(parts[1])
                    except ValueError:
                        params["neighbor"] = parts[1]
        except Exception:
            logger.debug("suppressed in _read_script_params", exc_info=True)
        return params

    def _try_autofix(
        self, input_path: Path, error: str
    ) -> dict[str, Any] | None:
        """跑一次 AutoFixLoop, 命中规则就改 input.lammps 返回修了啥. 没命中返回 None."""
        try:
            from huginn.execution.autofix import AutoFixLoop

            current = self._read_script_params(input_path)
            fixed = AutoFixLoop().apply_fix("lammps_tool", error, current)
            if not fixed:
                return None
            reasoning = fixed.pop("__auto_fix", None)
            fixed.pop("__auto_fix_patterns_matched", None)
            if not fixed:
                return None
            # _apply_script_fixes 会整行替换, 只喂实际变化的参数,
            # 避免把无关命令行 (如 'neighbor 2.0 bin') 重写丢参数
            changed = {k: v for k, v in fixed.items() if current.get(k) != v}
            if not changed:
                return None
            str_fixes = {k: str(v) for k, v in changed.items()}
            new_script = self._apply_script_fixes(
                input_path.read_text(encoding="utf-8"), str_fixes
            )
            input_path.write_text(new_script, encoding="utf-8")
            return {"fixes": changed, "reasoning": reasoning}
        except Exception:
            return None

    def _is_float(self, s: str) -> bool:
        try:
            float(s)
            return True
        except ValueError:
            return False

    def _to_float_or_str(self, s: str):
        try:
            return float(s)
        except ValueError:
            return s

    def parse_trajectory(self, traj_path: str | Path) -> dict[str, Any]:
        """Parse LAMMPS trajectory file and compute basic analyses.

        Supports .lammpstrj and .dump formats.
        Uses a Rust accelerator if available, falling back to pure Python.
        """
        from pathlib import Path

        traj_path = Path(traj_path)
        if not traj_path.exists():
            return {"error": "Trajectory file not found"}

        # Try Rust-accelerated parser first.
        if _HAS_HUGINN_EXT:
            try:
                result = huginn_ext.parse_lammps_dump(
                    str(traj_path),
                    compute_msd=True,
                    compute_rdf=True,
                    rdf_bins=100,
                    rdf_r_max=None,
                    include_frames=False,
                )
                if "error" not in result:
                    return result
            except Exception:
                logger.debug("suppressed in parse_trajectory", exc_info=True)

        return self._parse_trajectory_python(traj_path)

    def _parse_trajectory_python(self, traj_path: str | Path) -> dict[str, Any]:
        """Pure-Python LAMMPS trajectory parser (baseline/fallback)."""
        from pathlib import Path

        traj_path = Path(traj_path)

        result = {
            "n_frames": 0,
            "n_atoms": 0,
            "atom_types": set(),
            "box_bounds": [],
            "timesteps": [],
        }

        try:
            frames = []
            current_frame = None

            with traj_path.open("r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if line == "ITEM: TIMESTEP":
                    if current_frame:
                        frames.append(current_frame)
                    current_frame = {"atoms": []}
                    i += 1
                    if i < len(lines):
                        current_frame["timestep"] = int(lines[i].strip())
                        result["timesteps"].append(current_frame["timestep"])
                elif line.startswith("ITEM: NUMBER OF ATOMS"):
                    i += 1
                    if i < len(lines):
                        current_frame["n_atoms"] = int(lines[i].strip())
                        result["n_atoms"] = current_frame["n_atoms"]
                elif line.startswith("ITEM: BOX BOUNDS"):
                    bounds = []
                    for _ in range(3):
                        i += 1
                        if i < len(lines):
                            bounds.append([float(x) for x in lines[i].strip().split()])
                    current_frame["box"] = bounds
                    if not result["box_bounds"]:
                        result["box_bounds"] = bounds
                elif line.startswith("ITEM: ATOMS"):
                    # Parse atom data
                    atom_headers = line.replace("ITEM: ATOMS ", "").split()
                    atoms = []
                    for _ in range(current_frame.get("n_atoms", 0)):
                        i += 1
                        if i < len(lines):
                            parts = lines[i].strip().split()
                            atom = {}
                            for h, p in zip(atom_headers, parts):
                                try:
                                    atom[h] = float(p)
                                except ValueError:
                                    atom[h] = p
                            atoms.append(atom)
                            if "type" in atom:
                                result["atom_types"].add(int(atom["type"]))
                    current_frame["atoms"] = atoms
                i += 1

            if current_frame:
                frames.append(current_frame)

            result["n_frames"] = len(frames)
            result["atom_types"] = sorted(result["atom_types"])

            # Compute MSD if positions available
            if frames and len(frames) > 1 and all("x" in a for a in frames[0]["atoms"]):
                msd = self._compute_msd(frames)
                if msd:
                    result["msd"] = msd

            # Compute RDF if 2+ frames
            if (
                frames
                and len(frames) >= 1
                and all("x" in a for a in frames[0]["atoms"])
            ):
                rdf = self._compute_rdf(frames[-1])
                if rdf:
                    result["rdf"] = rdf

        except Exception as e:
            result["error"] = str(e)

        return result

    def _compute_msd(self, frames: list[dict]) -> list[dict] | None:
        """Compute mean squared displacement across frames."""
        try:
            msd_data = []
            ref_positions = []
            for atom in frames[0]["atoms"]:
                ref_positions.append([atom["x"], atom["y"], atom["z"]])

            for frame in frames[1:]:
                displacements = []
                for i, atom in enumerate(frame["atoms"]):
                    dx = atom["x"] - ref_positions[i][0]
                    dy = atom["y"] - ref_positions[i][1]
                    dz = atom["z"] - ref_positions[i][2]
                    displacements.append(dx * dx + dy * dy + dz * dz)
                msd = sum(displacements) / len(displacements)
                msd_data.append(
                    {
                        "timestep": frame.get("timestep", 0),
                        "msd": msd,
                    }
                )
            return msd_data
        except Exception:
            return None

    def _compute_rdf(
        self, frame: dict, bins: int = 100, r_max: float | None = None
    ) -> dict | None:
        """Compute radial distribution function for a single frame."""
        try:
            import numpy as np

            atoms = frame["atoms"]
            pos = np.array([[a["x"], a["y"], a["z"]] for a in atoms], dtype=np.float64)
            n = len(pos)

            # Estimate r_max from box
            box = frame.get("box", [[0, 10], [0, 10], [0, 10]])
            lx = box[0][1] - box[0][0]
            ly = box[1][1] - box[1][0]
            lz = box[2][1] - box[2][0]
            if r_max is None:
                r_max = min(lx, ly, lz) / 2

            dr = r_max / bins
            box_vec = np.array([lx, ly, lz])

            # Compute pairwise distances with minimum image convention.
            # ponytail: O(n²) memory — fine for n<10k, chunk for larger systems.
            d = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]  # (n, n, 3)
            d -= box_vec * np.round(d / box_vec)
            r = np.sqrt((d ** 2).sum(axis=2))  # (n, n)
            mask = (r > 0) & (r < r_max)
            distances = r[mask]

            # Bin distances — full matrix counts each pair twice (i,j)+(j,i).
            g, _ = np.histogram(distances, bins=bins, range=(0, r_max))
            g = g.astype(np.float64)

            # Normalize
            volume = lx * ly * lz
            rho = n / volume
            r_edges = np.linspace(0, r_max, bins + 1)
            r_inner = r_edges[:-1]
            r_outer = r_edges[1:]
            shell_vol = (4.0 / 3.0) * np.pi * (r_outer**3 - r_inner**3)
            shell_vol = np.where(shell_vol > 0, shell_vol, 1.0)
            g /= n * rho * shell_vol

            r_values = ((r_edges[:-1] + r_edges[1:]) / 2).tolist()
            return {"r": r_values, "g": g.tolist(), "bins": bins, "r_max": r_max}
        except Exception:
            return None

    # ------------------------------------------------------------------ DEM

    async def _handle_dem_packing(
        self, args: LammpsToolInput, context: ToolContext
    ) -> ToolResult:
        """DEM 颗粒碰撞模拟: 生成 LAMMPS granular 输入脚本 + 可选执行."""
        script = self._generate_dem_input_script(args)

        # 写脚本到工作目录
        if args.working_dir:
            work_dir = Path(args.working_dir)
        else:
            work_dir = Path(context.workspace) / f"lammps_dem_{args.output_prefix}"
        work_dir.mkdir(parents=True, exist_ok=True)
        script_path = work_dir / "input.dem.lammps"
        script_path.write_text(script, encoding="utf-8")

        data = {
            "action": "dem_packing",
            "script_path": str(script_path),
            "script_content": script,
            "n_particles": args.dem_n_particles,
            "particle_radius": args.dem_radius,
            "box_size": args.dem_box,
            "contact_model": "hertz/material",
            "n_steps": args.dem_n_steps,
            "friction_coeff": args.dem_friction,
            "restitution": args.dem_restitution,
        }

        # 没有 LAMMPS 可执行文件就只返回脚本
        if not self.lammps_executable:
            from huginn.tools.sim.executable_resolver import resolve_executable

            resolution = resolve_executable("lammps")
            if isinstance(resolution, str):
                self.lammps_executable = resolution
            else:
                data["needs_resolution"] = True
                data["resolution_request"] = resolution.to_dict()
                return ToolResult(data=data, success=True,
                                  error="LAMMPS executable not found. Script generated only.")

        # 执行 LAMMPS
        try:
            import subprocess

            cmd = [str(self.lammps_executable), "-in", str(script_path)]
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=3600, cwd=str(work_dir)
            )
            ok = proc.returncode == 0
            data["returncode"] = proc.returncode
            data["stdout_tail"] = proc.stdout[-2000:] if proc.stdout else ""
            data["stderr_tail"] = proc.stderr[-2000:] if proc.stderr else ""

            if ok:
                data["output_dir"] = str(work_dir)
                data["status"] = "completed"

                # provenance
                try:
                    from huginn.provenance import capture
                    data["provenance"] = capture(
                        "lammps_tool", args.model_dump(), output=dict(data)
                    ).to_dict()
                except Exception:
                    logger.debug("DEM provenance failed", exc_info=True)
            else:
                data["status"] = "failed"

            return ToolResult(
                data=data,
                success=ok,
                error=None if ok else f"LAMMPS exited with code {proc.returncode}",
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                data=data, success=False, error="LAMMPS DEM execution timed out (3600s)"
            )
        except Exception as e:
            return ToolResult(
                data=data, success=False, error=f"DEM execution failed: {e}"
            )

    @staticmethod
    def _generate_dem_input_script(args: LammpsToolInput) -> str:
        """生成 LAMMPS DEM (Discrete Element Method) 颗粒碰撞脚本.

        用 Hertz-Mindlin 接触模型 (LAMMPS pair_style granular):
        - 法向: Hertzian (k_n ∝ E* / R*)
        - 切向: Mindlin (k_t = (2-v)/(2(1-v)) * k_n)
        - 摩擦: 库仑摩擦 (μ)
        - 恢复: 非线性阻尼 (由 restitution coefficient 反算)

        LAMMPS GRANULAR pair style 需要编译 GRANULAR package.
        """
        bx, by, bz = args.dem_box
        r = args.dem_radius
        r_std = args.dem_radius_std
        rho = args.dem_density
        E = args.dem_youngs
        nu = args.dem_poisson
        mu = args.dem_friction
        e = args.dem_restitution
        n_steps = args.dem_n_steps
        g = args.dem_gravity

        # 颗粒质量: m = ρ * (4/3)πr³
        mass = rho * (4.0 / 3.0) * 3.14159265358979 * r ** 3

        # Hertzian 法向刚度: k_n = (4/3) * E* * sqrt(R*)
        # ponytail: LAMMPS 的 granular pair style 内部自己算, 这里只给 E 和 v
        # G = E / (2(1+v)) — shear modulus for Mindlin
        G = E / (2 * (1 + nu))

        # 用 SI 单位, 因为 DEM 颗粒通常在 mm~cm 尺度
        # 如果用户用 real 单位, 数值需要自己换算

        return f"""# LAMMPS DEM (Discrete Element Method) — Granular Packing Simulation
# 碰撞模型: Hertz-Mindlin with Coulomb friction
# 生成方式: lammps_tool action=dem_packing
# 粒子数: {args.dem_n_particles}, 粒径: {r} ± {r_std}

# ── Units & Atom Style ──────────────────────────────────────────
# si: meters/seconds/kg; real: Angstroms/fs/g (需按比例换算)
units           si
atom_style      sphere
boundary        f f f

# ── Simulation Box ──────────────────────────────────────────────
region          box block 0 {bx} 0 {by} 0 {bz}
create_box      1 box

# ── Particle Properties ─────────────────────────────────────────
# 每个粒子: position + diameter + density (LAMMPS 算 mass)
set             type 1 diameter {2*r} density {rho}

# ── Create Particles ────────────────────────────────────────────
# random distribution in box, 上下留点空给重力沉降
create_atoms    1 random {args.dem_n_particles} 12345 box \\
                overlap {2*r} maxtry 10000

# 多分散: 按 r ± r_std 调整粒径
{"variable       r_var normal " + str(r) + " " + str(r_std) if r_std > 0 else ""}
{"set             type 1 diameter v_r_var" if r_std > 0 else ""}

# ── Neighbor & Communication ───────────────────────────────────
neighbor        {r * 2} bin
neigh_modify    delay 0

# ── Pair Style: Hertz-Mindlin Granular Contact ──────────────────
# LAMMPS GRANULAR package: 非线性 Hertz 接触 + Mindlin 切向 + 阻尼
pair_style      granular
pair_coeff      * * hertz/material {E} {nu} {mu} {e} 0 normal \\
                mindlin/force {G} {mu} 0 tangential \\
                damping_coeff {e} 0.5 0.5 0 rolling \\
                tsudi 1.0 0.5 0 twisting

# ── Physics: Gravity (optional) ─────────────────────────────────
{"fix            gravity all gravity {g} vector 0 0 -1" if g > 0 else "# no gravity"}

# ── Integration: NVE + Granular Temperature ────────────────────
# velocity limit 防止穿透时爆飞
fix             integrate all nve/sphere
fix             freeze_property all setforce 0 0 0

# ── Output: Thermo + Dump ──────────────────────────────────────
thermo          1000
thermo_style    custom step atoms ke pe etotal press
thermo_modify   lost warn

# 每 10000 步 dump 一次粒子位置和速度
dump            particles all custom 10000 dump.particles id type x y z vx vy vz \\
                radius mass
dump_modify     particles sort id

# ── Run ─────────────────────────────────────────────────────────
# 初始能量最小化 (消除重叠)
minimize        1e-6 1e-8 1000 10000

# 释放 freeze, 跑 DEM
unfix           freeze_property

timestep        {1e-6 if g > 0 else 1e-7}
run             {n_steps}

# ── Post: Compute Coordination Number & Packing Fraction ───────
compute         cn all contact/atom
compute         cn_avg all reduce ave c_cn
variable        phi equal count(all) * {mass} / (vol * {rho})
variable        mean_cn equal c_cn_avg

print           "Packing fraction (phi): ${{phi}}"
print           "Mean coordination number: ${{mean_cn}}"

write_restart   restart.dem
"""
