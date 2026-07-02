"""VASP DFT calculation tool.

Supports both real VASP execution (if available) and mock mode.
"""

from __future__ import annotations

import asyncio
import os
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


# submit_async 实际可以跑的计算类型
_COMPUTE_ACTIONS = ("relax", "scf", "band", "dos", "md", "phonon")


class VaspToolInput(BaseModel):
    action: Literal[
        "relax",
        "scf",
        "band",
        "dos",
        "md",
        "phonon",
        "submit_async",
        "poll_job",
        "wait_job",
    ] = Field(...)
    working_dir: str | None = Field(
        default=None,
        description="Directory containing POSCAR/INCAR/POTCAR/KPOINTS",
    )
    incar_overrides: dict = Field(
        default_factory=dict, description="Override specific INCAR tags"
    )
    queue: Literal["debug", "normal", "gpu"] = Field(default="normal")
    walltime_hours: int = Field(default=24, ge=1, le=168)
    # submit_async 专用: 指定实际跑哪种计算 (relax/scf/band/...)
    compute_action: Literal["relax", "scf", "band", "dos", "md", "phonon"] | None = (
        Field(
            default=None,
            description="For submit_async: which computation to run (relax/scf/band/dos/md/phonon)",
        )
    )
    # poll_job / wait_job 专用
    job_id: str | None = Field(
        default=None,
        description="For poll_job/wait_job: the job_id returned by submit_async",
    )
    # wait_job 专用: 最长等多久 (秒), 超时返回 status=running
    timeout: float = Field(
        default=3600.0,
        ge=1.0,
        description="For wait_job: max seconds to wait before returning (default 3600)",
    )
    # 计算失败时自动诊断 + 改 INCAR 重试的次数. 0 = 关闭自愈.
    max_auto_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description="On failure, auto-diagnose + patch INCAR and retry up to N times",
    )

    @model_validator(mode="after")
    def _check_action_fields(self) -> "VaspToolInput":
        """不同 action 需要不同字段, 在 schema 层兜底, 别等 call() 才挂."""
        if self.action in _COMPUTE_ACTIONS or self.action == "submit_async":
            if not self.working_dir:
                raise ValueError(
                    f"action '{self.action}' requires 'working_dir'"
                )
        if self.action == "submit_async" and not self.compute_action:
            raise ValueError(
                "submit_async requires 'compute_action' (relax/scf/band/dos/md/phonon)"
            )
        if self.action in ("poll_job", "wait_job") and not self.job_id:
            raise ValueError(f"action '{self.action}' requires 'job_id'")
        return self


class VaspToolOutput(BaseModel):
    job_id: str | None = None
    status: Literal["completed", "failed", "mock"] = "mock"
    energy: float | None = None
    converged: bool = False
    output_files: list[str] = []
    warnings: list[str] = []


class VaspTool(HuginnTool):
    """Submit and manage VASP DFT calculations."""

    name = "vasp_tool"
    category = "sim"
    profile = ToolProfile(
        cost_tier="heavy",
        phases=frozenset({ResearchPhase.EXECUTION}),
        constraint_scope="dft",
        light_alternatives=(
            "materials_database_tool",
            "local_structure_db",
            "symbolic_math_tool",
            "numerical_tool",
        ),
    )
    description = (
        "Run VASP DFT calculations (relaxation, SCF, band structure, DOS, MD, phonons). "
        "Supports async submission via submit_async / poll_job / wait_job for long-running jobs."
    )
    input_schema = VaspToolInput
    _init_kwargs_map = {"vasp_executable": "vasp_executable"}

    # 异步作业注册表: job_id -> {status, task, result, error, started_at, finished_at}
    # 类级别共享, 同一进程内多次 submit_async / poll_job 能互通.
    # 注意: 进程重启后作业状态丢失, 长跑作业建议用 job_tool 走 HPC.
    _async_jobs: dict[str, dict[str, Any]] = {}

    def __init__(
        self, vasp_executable: str | None = None, sandbox: SandboxExecutor | None = None
    ):
        super().__init__()
        self.vasp_executable = vasp_executable or self._find_vasp()
        self.sandbox = sandbox or SandboxExecutor()

    def _find_vasp(self) -> str | None:
        """Find VASP executable."""
        env_path = os.environ.get("VASP_EXECUTABLE")
        if env_path and Path(env_path).exists():
            return env_path

        # Check PATH
        try:
            import shutil

            for name in ["vasp", "vasp_std", "vasp_gam", "vasp_ncl"]:
                exe = shutil.which(name)
                if exe:
                    return exe
        except Exception:
            pass

        return None

    def estimate_cost(self, args: VaspToolInput) -> dict[str, float] | None:
        # poll_job / wait_job 是查询操作, 不消耗计算资源
        if args.action in ("poll_job", "wait_job"):
            return None
        return {
            "cpu_hours": args.walltime_hours * 4,
            "walltime_hours": args.walltime_hours,
        }

    async def validate_input(
        self, args: VaspToolInput, context: ToolContext
    ) -> ValidationResult:
        """Pre-flight: verify working directory and required VASP input files.

        submit_async 跟普通计算 action 一样需要 working_dir + POSCAR.
        poll_job / wait_job 只需要 job_id, 不检查工作目录.
        """
        if args.action in ("poll_job", "wait_job"):
            # job_id 在 schema 层已经强制非空, 这里直接放行
            return ValidationResult(result=True)

        vr = HandleValidator.validate(HandleType.FILE_PATH, args.working_dir, context)
        if not vr.result:
            return ValidationResult(
                result=False,
                message=f"Working directory not found: {args.working_dir}",
                error_code=404,
            )
        poscar = Path(args.working_dir) / "POSCAR"
        if not poscar.exists():
            ws_poscar = Path(context.workspace) / args.working_dir / "POSCAR" if context.workspace else None
            if not (ws_poscar and ws_poscar.exists()):
                return ValidationResult(
                    result=False,
                    message="POSCAR not found in working directory",
                    error_code=404,
                )
        return ValidationResult(result=True)

    async def call(self, args: VaspToolInput, context: ToolContext) -> ToolResult:
        # 异步作业管理动作: 不跑实际计算, 只查/等作业状态
        if args.action == "submit_async":
            return await self._handle_submit_async(args, context)
        if args.action == "poll_job":
            return self._handle_poll_job(args)
        if args.action == "wait_job":
            return await self._handle_wait_job(args)

        work_dir = Path(args.working_dir)
        if not work_dir.exists():
            return ToolResult(
                data=None,
                success=False,
                error=f"Working directory not found: {work_dir}",
            )

        # Check for required input files
        poscar = work_dir / "POSCAR"
        incar = work_dir / "INCAR"
        if not poscar.exists():
            return ToolResult(
                data=None, success=False, error="POSCAR not found in working directory"
            )

        # Apply INCAR overrides
        if args.incar_overrides and incar.exists():
            self._modify_incar(incar, args.incar_overrides)

        # If VASP is available, run it
        if self.vasp_executable:
            return await self._run_vasp(args, work_dir)

        # Mock mode: return synthetic results
        return self._mock_result(args, work_dir)

    async def _run_vasp(self, args: VaspToolInput, work_dir: Path) -> ToolResult:
        """Execute real VASP calculation, 自动诊断+改 INCAR 重试."""
        autoheal_log: list[dict[str, Any]] = []
        try:
            cmd = [self.vasp_executable]
            result: Any = None
            max_retries = args.max_auto_retries

            for attempt in range(max_retries + 1):
                sb_result = self.sandbox.run(
                    cmd,
                    cwd=str(work_dir),
                    timeout=args.walltime_hours * 3600,
                    queue=args.queue,
                    walltime=f"{args.walltime_hours}:00:00",
                )
                result = sb_result

                if result.returncode == 0:
                    break
                # 失败了, 看还有没有重试额度 + 能不能自动修
                if attempt < max_retries:
                    fixed = self._try_autofix(work_dir, result.stderr or "")
                    if fixed:
                        autoheal_log.append(
                            {
                                "attempt": attempt + 1,
                                "error": (result.stderr or "")[:300],
                                "fixes_applied": fixed["fixes"],
                                "reasoning": fixed["reasoning"],
                            }
                        )
                        continue
                break  # 没修动或重试耗尽

            # Parse OUTCAR for comprehensive results
            outcar = work_dir / "OUTCAR"
            parsed = self._parse_outcar(outcar) if outcar.exists() else {}

            # Also try vasprun.xml for structured data
            vasprun = work_dir / "vasprun.xml"
            if vasprun.exists():
                parsed.update(self._parse_vasprun_quick(vasprun))

            output = VaspToolOutput(
                status="completed" if result.returncode == 0 else "failed",
                energy=parsed.get("energy"),
                converged=parsed.get("converged", False),
                output_files=[
                    f.name
                    for f in work_dir.iterdir()
                    if f.suffix in [".OUTCAR", ".vasprun", ".CHG"]
                ],
            )

            # Include parsed details in result
            data = output.model_dump()
            data["parsed"] = parsed
            if autoheal_log:
                data["autoheal_attempts"] = autoheal_log

            # 带上 provenance 快照, 事后能追溯参数/版本/环境
            try:
                from huginn.provenance import capture

                data["provenance"] = capture(
                    "vasp_tool", args.model_dump(), output=dict(data)
                ).to_dict()
            except Exception:
                pass  # provenance 失败不能把计算结果带挂

            return ToolResult(
                data=data,
                success=result.returncode == 0,
                error=result.stderr[:500] if result.returncode != 0 else None,
            )

        except subprocess.TimeoutExpired:
            return ToolResult(
                data=None,
                success=False,
                error=f"VASP execution timed out ({args.walltime_hours}h)",
            )
        except Exception as e:
            return ToolResult(
                data=None, success=False, error=f"VASP execution failed: {e}"
            )

    def _read_incar_params(self, work_dir: Path) -> dict[str, Any]:
        """读 INCAR 解析成 dict, 给 AutoFixLoop 判断当前参数用."""
        incar = work_dir / "INCAR"
        if not incar.exists():
            return {}
        params: dict[str, Any] = {}
        try:
            for line in incar.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                if "=" not in s:
                    continue
                k, v = s.split("=", 1)
                k = k.strip().upper()
                v = v.strip()
                # 数字尽量转成数字, 方便 halve/double 规则
                try:
                    num = float(v)
                    v = int(num) if num == int(num) else num  # type: ignore[assignment]
                except ValueError:
                    pass
                params[k] = v
        except Exception:
            pass
        return params

    def _try_autofix(self, work_dir: Path, stderr: str) -> dict[str, Any] | None:
        """跑一次 AutoFixLoop, 命中规则就改 INCAR 返回修了啥. 没命中返回 None."""
        incar = work_dir / "INCAR"
        if not incar.exists():
            return None
        try:
            from huginn.execution.autofix import AutoFixLoop

            current = self._read_incar_params(work_dir)
            fixed = AutoFixLoop().apply_fix("vasp_tool", stderr, current)
            if not fixed:
                return None
            reasoning = fixed.pop("__auto_fix", None)
            fixed.pop("__auto_fix_patterns_matched", None)
            # 剩下的才是真正要改的 INCAR tag
            if not fixed:
                return None
            self._modify_incar(incar, fixed)
            return {"fixes": fixed, "reasoning": reasoning}
        except Exception:
            return None

    # ------------------------------------------------------------------ async job API

    async def _handle_submit_async(
        self, args: VaspToolInput, context: ToolContext
    ) -> ToolResult:
        """异步提交 VASP 计算, 立即返回 job_id, 不阻塞等待结果.

        内部用 asyncio.create_task 在后台跑实际计算 (relax/scf/band/...),
        计算完成后把结果存进 _async_jobs[job_id]. 进程重启后作业状态丢失,
        长跑作业建议走 job_tool 提交到 HPC.

        Returns:
            ToolResult.data = {"job_id": str, "status": "running", "compute_action": str}
        """
        # 构造同步调用的 args: 用 compute_action 作为 action
        sync_args = VaspToolInput(
            action=args.compute_action,
            working_dir=args.working_dir,
            incar_overrides=args.incar_overrides,
            queue=args.queue,
            walltime_hours=args.walltime_hours,
        )

        job_id = f"vasp-{uuid.uuid4().hex[:12]}"

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
        VaspTool._async_jobs[job_id] = job_entry

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

    def _handle_poll_job(self, args: VaspToolInput) -> ToolResult:
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
        job = VaspTool._async_jobs.get(job_id)
        if job is None:
            return ToolResult(
                data=None,
                success=False,
                error=f"Unknown job_id: {job_id}",
            )

        elapsed = time.time() - job["started_at"]
        status = job["status"]
        # 进度估算: running 给 50 (无法精确跟踪 VASP 内部进度),
        # done/failed 给 100. 真要精确进度得解析 OUTCAR 的 ionic step,
        # 这里先做粗略估计.
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

    async def _handle_wait_job(self, args: VaspToolInput) -> ToolResult:
        """阻塞等待作业完成或超时.

        内部用 asyncio.wait_for 等后台 task, 超时返回当前状态 (status=running).
        作业完成返回最终结果, 失败返回错误.
        """
        job_id = args.job_id
        timeout = args.timeout
        job = VaspTool._async_jobs.get(job_id)
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

    def _parse_outcar(self, outcar_path: Path) -> dict[str, Any]:
        """Parse OUTCAR for key physical quantities.

        Uses the Rust accelerator when available and falls back to pure Python.
        """
        if _HAS_HUGINN_EXT:
            try:
                result = huginn_ext.parse_outcar(str(outcar_path))
                if "error" not in result:
                    return result
            except Exception:
                pass

        return self._parse_outcar_python(outcar_path)

    def _parse_outcar_python(self, outcar_path: Path) -> dict[str, Any]:
        """Pure-Python OUTCAR parser (baseline/fallback)."""
        import re

        result = {
            "energy": None,
            "converged": False,
            "forces": [],
            "magnetic_moments": [],
            "lattice_vectors": [],
            "volume": None,
            "band_gap": None,
            "encut": None,
            "kpoints": None,
            "nelm": None,
            "nelmin": None,
            "ispin": None,
        }

        # 优先用 pymatgen 解析, 拿不到的字段留给后面的 regex 兜底
        try:
            from pymatgen.io.vasp import Outcar

            oc = Outcar(str(outcar_path))
            if oc.final_energy is not None:
                result["energy"] = float(oc.final_energy)
            if oc.forces:
                result["forces"] = [
                    {"position": [0.0, 0.0, 0.0], "force": list(f)}
                    for f in oc.forces[-1]
                ]
            if oc.magnetizations:
                result["magnetic_moments"] = list(oc.magnetizations[-1])
            result["converged"] = bool(oc.converged)
            result["parse_source"] = "pymatgen"
        except Exception:
            pass  # pymatgen 没装或解析失败, 走下面的 regex

        try:
            content = outcar_path.read_text(encoding="utf-8", errors="ignore")

            # Energy — pymatgen 已经填了就不覆盖
            if result["energy"] is None:
                energy_matches = re.findall(r"free  energy   TOTEN  =\s+([-\d.]+)", content)
                if energy_matches:
                    result["energy"] = float(energy_matches[-1])

            # Convergence — pymatgen 的更可靠, 没填才用启发式
            if not result["converged"]:
                result["converged"] = "reached required accuracy" in content

            # ENCUT
            encut_match = re.search(r"ENCUT\s*=\s*([\d.]+)", content)
            if encut_match:
                result["encut"] = float(encut_match.group(1))

            # ISPIN
            ispin_match = re.search(r"ISPIN\s*=\s*(\d+)", content)
            if ispin_match:
                result["ispin"] = int(ispin_match.group(1))

            # NELM / NELMIN
            nelm_match = re.search(r"NELM\s*=\s*(\d+)", content)
            if nelm_match:
                result["nelm"] = int(nelm_match.group(1))
            nelmin_match = re.search(r"NELMIN\s*=\s*(\d+)", content)
            if nelmin_match:
                result["nelmin"] = int(nelmin_match.group(1))

            # K-points
            kpoint_match = re.search(
                r"k-points in units of 2pi/SCALE and weight:.*\n.*\n.*", content
            )
            if kpoint_match:
                result["kpoints"] = "found"  # Simplified

            # Lattice vectors (last occurrence)
            lattice_pattern = r"direct lattice vectors\s+reciprocal lattice vectors\n\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+.*\n\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+.*\n\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+.*"
            lattice_matches = re.findall(lattice_pattern, content)
            if lattice_matches:
                last = lattice_matches[-1]
                result["lattice_vectors"] = [
                    [float(last[0]), float(last[1]), float(last[2])],
                    [float(last[3]), float(last[4]), float(last[5])],
                    [float(last[6]), float(last[7]), float(last[8])],
                ]

            # Volume
            vol_match = re.findall(r"volume of cell :\s+([\d.]+)", content)
            if vol_match:
                result["volume"] = float(vol_match[-1])

            # Final forces — pymatgen 已填就不覆盖
            if not result["forces"]:
                force_section = re.findall(
                    r"TOTAL-FORCE.*?\n(.*?)(?:\n\n|\n---)", content, re.DOTALL
                )
                if force_section:
                    forces = []
                    for line in force_section[-1].strip().split("\n"):
                        parts = line.split()
                        if len(parts) >= 6 and all(self._is_float(p) for p in parts[:6]):
                            forces.append(
                                {
                                    "position": [
                                        float(parts[0]),
                                        float(parts[1]),
                                        float(parts[2]),
                                    ],
                                    "force": [
                                        float(parts[3]),
                                        float(parts[4]),
                                        float(parts[5]),
                                    ],
                                }
                            )
                    result["forces"] = forces

            # Magnetic moments — pymatgen 已填就不覆盖
            if not result["magnetic_moments"]:
                mag_matches = re.findall(
                    r"magnetization \(x\).*?\n(.*?)(?:\n\n|\n---)", content, re.DOTALL
                )
                if mag_matches:
                    mag_moments = []
                    for line in mag_matches[-1].strip().split("\n"):
                        parts = line.split()
                        if len(parts) >= 5 and self._is_float(parts[-1]):
                            mag_moments.append(float(parts[-1]))
                    result["magnetic_moments"] = mag_moments

            # Band gap — OUTCAR 里没有直接值, 只记 efermi, 真值靠 vasprun.xml
            efermi_match = re.search(r"E-fermi\s*:\s*([-\d.]+)", content)
            if efermi_match:
                result["efermi"] = float(efermi_match.group(1))

        except Exception as e:
            result["parse_error"] = str(e)

        return result

    def _parse_vasprun_quick(self, vasprun_path: Path) -> dict[str, Any]:
        """Quick-parse vasprun.xml for structured data."""
        import xml.etree.ElementTree as ET

        result = {"parse_source": "vasprun.xml"}

        # 优先用 pymatgen 拿 band gap / efermi, 失败落回 ElementTree
        try:
            from pymatgen.io.vasp import Vasprun

            vr = Vasprun(str(vasprun_path))
            try:
                gap, cbm, vbm = vr.eigenvalue_band_properties
                result["band_gap"] = float(gap) if gap is not None else None
                result["cbm"] = float(cbm) if cbm is not None else None
                result["vbm"] = float(vbm) if vbm is not None else None
            except Exception:
                pass
            result["efermi"] = float(vr.efermi) if vr.efermi is not None else None
            result["parse_source"] = "pymatgen_vasprun"
            return result
        except Exception:
            pass  # pymatgen 没装或解析失败, 走 ElementTree

        try:
            tree = ET.parse(vasprun_path)
            root = tree.getroot()

            # Find calculation/energy/i
            for calc in root.findall(".//calculation"):
                energy_elem = calc.find(".//energy/i[@name='e_wo_entrp']")
                if energy_elem is not None and energy_elem.text:
                    result["energy_vasprun"] = float(energy_elem.text)

                # Forces
                varray = calc.find(".//varray[@name='forces']")
                if varray is not None:
                    forces = []
                    for v in varray.findall("v"):
                        forces.append([float(x) for x in v.text.split()])
                    result["forces_vasprun"] = forces
                break  # Only first calc for quick parse

            # K-points
            kpoints = root.find(".//kpoints")
            if kpoints is not None:
                varray = kpoints.find("varray[@name='kpointlist']")
                if varray is not None:
                    result["kpoint_count"] = len(varray.findall("v"))

        except Exception as e:
            result["parse_error"] = str(e)

        return result

    def _is_float(self, s: str) -> bool:
        try:
            float(s)
            return True
        except ValueError:
            return False

    def _mock_result(self, args: VaspToolInput, work_dir: Path) -> ToolResult:
        """Generate mock results when VASP is not available."""
        import random

        mock_energies = {
            "relax": -150.0,
            "scf": -152.3,
            "band": -152.3,
            "dos": -152.3,
            "md": -148.5,
            "phonon": -152.3,
        }

        output = VaspToolOutput(
            status="mock",
            energy=mock_energies.get(args.action, -100.0) + random.uniform(-0.5, 0.5),
            converged=True,
            output_files=["OUTCAR", "vasprun.xml", "OSZICAR"],
            warnings=[
                "VASP executable not found. Results are MOCK data for demonstration."
            ],
        )

        data = output.model_dump()
        # mock 数据也带 provenance, 方便区分真跑 vs 演示
        try:
            from huginn.provenance import capture

            data["provenance"] = capture(
                "vasp_tool", args.model_dump(), output=dict(data)
            ).to_dict()
        except Exception:
            pass

        return ToolResult(
            data=data,
            success=True,
            error=None,
        )

    def _modify_incar(self, incar_path: Path, overrides: dict) -> None:
        """Modify INCAR file with override values."""
        try:
            content = incar_path.read_text(encoding="utf-8")
            lines = content.split("\n")
            modified = []
            overridden_keys = set()

            for line in lines:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    modified.append(line)
                    continue

                # Check if this line defines a key we want to override
                for key in overrides:
                    if stripped.upper().startswith(
                        key.upper() + " ="
                    ) or stripped.upper().startswith(key.upper() + "="):
                        modified.append(f"{key} = {overrides[key]}")
                        overridden_keys.add(key)
                        break
                else:
                    modified.append(line)

            # Add any new keys that weren't in the original file
            for key, value in overrides.items():
                if key not in overridden_keys:
                    modified.append(f"{key} = {value}")

            incar_path.write_text("\n".join(modified), encoding="utf-8")

        except Exception as e:
            print(f"Warning: Failed to modify INCAR: {e}")
