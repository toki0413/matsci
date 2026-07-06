"""Quantum ESPRESSO DFT tool — generate pw.x input and parse output.

When QE is not installed, the tool falls back to input-export mode.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.security import SandboxConfig, SandboxExecutor
from huginn.tools.base import HuginnTool, ResearchPhase, ToolProfile
from huginn.types import ToolContext, ToolResult


class QuantumEspressoToolInput(BaseModel):
    action: Literal["generate", "run", "parse"] = Field(default="run")
    calculation: Literal["scf", "relax", "vc-relax", "md", "bands"] = Field(
        default="scf"
    )
    structure: dict = Field(
        default_factory=lambda: {
            "lattice": [[5.43, 0.0, 0.0], [0.0, 5.43, 0.0], [0.0, 0.0, 5.43]],
            "species": ["Si", "Si"],
            "positions": [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
            "units": "crystal",
        }
    )
    pseudopotentials: dict[str, str] = Field(
        default_factory=lambda: {"Si": "Si.pbe-n-kjpaw_psl.1.0.0.UPF"}
    )
    kpoints: dict = Field(
        default_factory=lambda: {"mode": "automatic", "grid": [4, 4, 4]}
    )
    ecutwfc: float = Field(default=40.0)
    ecutrho: float | None = Field(default=None)
    smearing: str = Field(default="gaussian")
    degauss: float = Field(default=0.01)
    nspin: Literal[1, 2] = Field(default=1)
    mixing_beta: float = Field(default=0.7)
    electron_maxstep: int = Field(default=100)
    working_dir: str | None = Field(default=None)
    output_prefix: str = Field(default="qe_out")
    result_files: list[str] = Field(default_factory=list)
    # 计算失败 / SCF 没收敛时自动诊断 + 改输入重试的次数. 0 = 关闭自愈.
    max_auto_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description="On failure or non-convergence, auto-diagnose + patch input and retry up to N times",
    )


class QuantumEspressoTool(HuginnTool):
    """Generate and run Quantum ESPRESSO pw.x calculations."""

    name = "qe_tool"
    category = "sim"
    profile = ToolProfile(
        cost_tier="heavy",
        phases=frozenset({ResearchPhase.EXECUTION}),
        constraint_scope="dft",
        light_alternatives=(
            "materials_database_tool",
            "local_structure_db",
            "symbolic_math_tool",
        ),
    )
    description = (
        "Generate and run Quantum ESPRESSO DFT calculations. "
        "Falls back to exporting the input file when pw.x is not installed."
    )
    input_schema = QuantumEspressoToolInput

    def __init__(
        self, qe_executable: str | None = None, sandbox: SandboxExecutor | None = None
    ):
        super().__init__()
        self.qe_executable = qe_executable or self._find_qe()
        self.sandbox = sandbox or SandboxExecutor()

    def _find_qe(self) -> str | None:
        env_path = os.environ.get("QE_EXECUTABLE")
        if env_path and Path(env_path).exists():
            return env_path
        for cmd in ["pw.x", "pw"]:
            if shutil.which(cmd):
                return cmd
        return None

    def call(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        input_data = QuantumEspressoToolInput(**args)
        work_dir = (
            Path(input_data.working_dir) if input_data.working_dir else Path.cwd()
        )
        work_dir.mkdir(parents=True, exist_ok=True)

        try:
            if input_data.action == "parse":
                return self._parse_results(input_data, work_dir)

            input_path = self._generate_input(
                input_data, work_dir, input_data.output_prefix
            )

            if input_data.action == "generate":
                return ToolResult(
                    data={
                        "input_path": str(input_path),
                        "qe_available": self.qe_executable is not None,
                        "message": "Generated Quantum ESPRESSO input file.",
                    },
                    success=True,
                )

            return self._run_qe(input_data, work_dir, input_path)
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"QE tool failed: {e}")

    def _generate_input(
        self, args: QuantumEspressoToolInput, work_dir: Path, prefix: str
    ) -> Path:
        input_path = work_dir / f"{prefix}.in"
        ecutrho = args.ecutrho or 4 * args.ecutwfc

        lines = []
        lines.append("&CONTROL")
        lines.append(f"  calculation = '{args.calculation}'")
        lines.append(f"  prefix = '{prefix}'")
        lines.append(f"  outdir = '{work_dir.as_posix()}'")
        lines.append("  pseudo_dir = './'")
        lines.append("  tprnfor = .true.")
        lines.append("  tstress = .true.")
        lines.append("/")

        lines.append("&SYSTEM")
        lines.append("  ibrav = 0")
        lines.append(f"  nat = {len(args.structure.get('species', []))}")
        lines.append(f"  ntyp = {len(set(args.structure.get('species', [])))}")
        lines.append(f"  ecutwfc = {args.ecutwfc}")
        lines.append(f"  ecutrho = {ecutrho}")
        lines.append(f"  smearing = '{args.smearing}'")
        lines.append(f"  degauss = {args.degauss}")
        lines.append(f"  nspin = {args.nspin}")
        lines.append("/")

        lines.append("&ELECTRONS")
        lines.append(f"  mixing_beta = {args.mixing_beta}")
        lines.append(f"  electron_maxstep = {args.electron_maxstep}")
        lines.append("/")

        if args.calculation in ("relax", "vc-relax", "md"):
            lines.append("&IONS")
            lines.append("/")

        if args.calculation == "vc-relax":
            lines.append("&CELL")
            lines.append("/")

        lines.append("ATOMIC_SPECIES")
        species = args.structure.get("species", [])
        for element in sorted(set(species)):
            pseudo = args.pseudopotentials.get(element, f"{element}.UPF")
            mass = self._atomic_mass(element)
            lines.append(f"  {element} {mass:.4f} {pseudo}")

        lines.append("ATOMIC_POSITIONS {angstrom}")
        positions = args.structure.get("positions", [])
        for elem, pos in zip(species, positions):
            lines.append(f"  {elem} {pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}")

        lines.append("K_POINTS {automatic}")
        grid = args.kpoints.get("grid", [4, 4, 4])
        shift = args.kpoints.get("shift", [0, 0, 0])
        lines.append(
            f"  {grid[0]} {grid[1]} {grid[2]} {shift[0]} {shift[1]} {shift[2]}"
        )

        lines.append("CELL_PARAMETERS {angstrom}")
        lattice = args.structure.get("lattice", [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        for row in lattice:
            lines.append(f"  {row[0]:.8f} {row[1]:.8f} {row[2]:.8f}")

        input_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return input_path

    def _atomic_mass(self, element: str) -> float:
        masses = {
            "H": 1.008,
            "He": 4.0026,
            "Li": 6.94,
            "Be": 9.0122,
            "B": 10.81,
            "C": 12.011,
            "N": 14.007,
            "O": 15.999,
            "F": 18.998,
            "Ne": 20.180,
            "Na": 22.990,
            "Mg": 24.305,
            "Al": 26.982,
            "Si": 28.085,
            "P": 30.974,
            "S": 32.06,
            "Cl": 35.45,
            "Ar": 39.948,
            "K": 39.098,
            "Ca": 40.078,
            "Sc": 44.956,
            "Ti": 47.867,
            "V": 50.942,
            "Cr": 51.996,
            "Mn": 54.938,
            "Fe": 55.845,
            "Co": 58.933,
            "Ni": 58.693,
            "Cu": 63.546,
            "Zn": 65.38,
        }
        return masses.get(element, 1.0)

    def _run_qe(
        self, args: QuantumEspressoToolInput, work_dir: Path, input_path: Path
    ) -> ToolResult:
        if not self.qe_executable:
            return ToolResult(
                data={
                    "input_path": str(input_path),
                    "qe_available": False,
                    "message": (
                        "QE executable not found. Input file exported; "
                        "run manually with: pw.x -in " + input_path.name
                    ),
                },
                success=True,
            )

        output_path = work_dir / f"{args.output_prefix}.out"
        cmd = [self.qe_executable, "-in", str(input_path)]

        cfg = SandboxConfig(dry_run=False)
        autoheal_log: list[dict[str, Any]] = []
        result: dict[str, Any] = {}
        # 软失败原因. returncode=0 但 SCF 没收敛属于软失败,
        # 这种情况 result returncode==0, 不能当成功返回.
        soft_failure_msg: str | None = None
        max_retries = args.max_auto_retries

        for attempt in range(max_retries + 1):
            with open(output_path, "w", encoding="utf-8") as stdout_file:
                result = self.sandbox.run(
                    cmd,
                    cwd=work_dir,
                    config=cfg,
                    stdout=stdout_file,
                    stderr=subprocess.STDOUT,
                )

            rc = result.get("returncode", -1)
            # 判断这次跑完到底算成功还是失败:
            # - rc != 0 → 硬失败, stderr 被合进 output 文件了, 读出来诊断
            # - rc == 0 但 SCF 没收敛 → 软失败, QE 经常静默返回 0
            error: str | None = None
            soft_failure_msg = None
            if rc != 0:
                error = self._read_output_tail(output_path)
            else:
                parsed_now = self._parse_output_file(output_path)
                if not parsed_now.get("converged", True):
                    error = "SCF did not converge — convergence not achieved"
                    soft_failure_msg = error

            if error is None:
                break  # 真正成功

            # 失败了 (硬失败或软失败), 看还有没有重试额度 + 能不能自动修
            if attempt < max_retries:
                fixed = self._try_autofix(input_path, error)
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

        parsed = self._parse_output_file(output_path)
        ok = result.get("returncode", -1) == 0 and soft_failure_msg is None
        data: dict[str, Any] = {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "qe_available": True,
            "parsed": parsed,
            "message": (
                "QE execution completed."
                if ok
                else "QE execution failed; see output."
            ),
        }
        if autoheal_log:
            data["autoheal_attempts"] = autoheal_log

        return ToolResult(
            data=data,
            success=ok,
            error=(
                None
                if ok
                else (soft_failure_msg or "QE execution failed; see output.")
            ),
        )

    def _read_output_tail(self, output_path: Path, tail: int = 2000) -> str:
        """Read the tail of the QE output for error diagnosis.

        QE 的 stderr 被 stdout=STDOUT 合进了 output 文件, 硬失败时从这读错误文本.
        """
        try:
            content = output_path.read_text(encoding="utf-8", errors="ignore")
            return content[-tail:]
        except Exception:
            return ""

    def _read_input_params(self, input_path: Path) -> dict[str, Any]:
        """读 QE .in 解析成 dict, 给 AutoFixLoop 判断当前参数用."""
        params: dict[str, Any] = {}
        try:
            for line in input_path.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if not s or s.startswith("!") or s.startswith("&") or s == "/":
                    continue
                if "=" not in s:
                    continue
                k, v = s.split("=", 1)
                k = k.strip().lower()
                v = v.strip().strip("'\"")
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

    def _apply_input_fixes(self, input_path: Path, fixes: dict[str, Any]) -> None:
        """Patch QE .in 的 &ELECTRONS 块 (conv_thr / mixing_beta / mixing_type ...)."""
        try:
            lines = input_path.read_text(encoding="utf-8").split("\n")
            # 找 &ELECTRONS 块边界, SCF 相关参数都在这
            block_start = block_end = None
            for i, line in enumerate(lines):
                s = line.strip().lower()
                if s.startswith("&electrons"):
                    block_start = i
                elif block_start is not None and s == "/":
                    block_end = i
                    break
            if block_start is None or block_end is None:
                return  # 没找到 &ELECTRONS, 不敢瞎改
            # 已有参数的行索引, 方便原地替换
            existing: dict[str, int] = {}
            for i in range(block_start + 1, block_end):
                if "=" in lines[i]:
                    key = lines[i].split("=")[0].strip().lower()
                    existing[key] = i
            for key, val in fixes.items():
                line_new = f"  {key} = {val}"
                if key.lower() in existing:
                    lines[existing[key.lower()]] = line_new
                else:
                    # 块里没有的参数, 插到闭合 / 之前
                    lines.insert(block_end, line_new)
                    block_end += 1
            input_path.write_text("\n".join(lines), encoding="utf-8")
        except Exception as e:
            print(f"Warning: failed to patch QE input: {e}")

    def _try_autofix(self, input_path: Path, error: str) -> dict[str, Any] | None:
        """跑一次 AutoFixLoop, 命中规则就改 QE 输入返回修了啥. 没命中返回 None."""
        try:
            from huginn.execution.autofix import AutoFixLoop

            current = self._read_input_params(input_path)
            fixed = AutoFixLoop().apply_fix("qe_tool", error, current)
            if not fixed:
                return None
            reasoning = fixed.pop("__auto_fix", None)
            fixed.pop("__auto_fix_patterns_matched", None)
            if not fixed:
                return None
            self._apply_input_fixes(input_path, fixed)
            return {"fixes": fixed, "reasoning": reasoning}
        except Exception:
            return None

    def _parse_output_file(self, output_path: Path) -> dict[str, Any]:
        if not output_path.exists():
            return {"error": "Output file not found"}

        content = output_path.read_text(encoding="utf-8", errors="ignore")
        return self._parse_output(content)

    def _parse_output(self, content: str) -> dict[str, Any]:
        result: dict[str, Any] = {
            "energy": None,
            "converged": False,
            "forces": [],
            "stress": [],
            "n_scf_steps": 0,
        }

        lines = content.splitlines()

        # Energy
        for line in lines:
            if "!    total energy" in line:
                parts = line.split("=")
                if len(parts) > 1:
                    with contextlib.suppress(ValueError, IndexError):
                        result["energy"] = float(parts[-1].strip().split()[0])

        # Convergence
        result["converged"] = "convergence has been achieved" in content

        # SCF steps
        result["n_scf_steps"] = content.count("iteration #")

        # Forces (last block)
        force_blocks = []
        current_block = []
        in_forces = False
        for line in lines:
            if "Forces acting on atoms" in line:
                in_forces = True
                current_block = []
                continue
            if in_forces:
                if line.strip().startswith("atom"):
                    parts = line.split()
                    if len(parts) >= 8:
                        with contextlib.suppress(ValueError, IndexError):
                            fx = float(parts[-3])
                            fy = float(parts[-2])
                            fz = float(parts[-1])
                            current_block.append([fx, fy, fz])
                elif "Total force" in line:
                    if current_block:
                        force_blocks.append(current_block)
                        current_block = []
                    in_forces = False
                elif line.strip() == "" and current_block:
                    force_blocks.append(current_block)
                    current_block = []
                    in_forces = False
        if current_block:
            force_blocks.append(current_block)
        if force_blocks:
            result["forces"] = force_blocks[-1]

        # Stress (last occurrence)
        for i, line in enumerate(lines):
            if (
                "total   stress" in line
                or "stress" in line.lower()
                and "kbar" in line.lower()
            ):
                stress = []
                for j in range(1, 4):
                    if i + j < len(lines):
                        parts = lines[i + j].split()
                        if len(parts) >= 3:
                            with contextlib.suppress(ValueError, IndexError):
                                stress.append(
                                    [float(parts[0]), float(parts[1]), float(parts[2])]
                                )
                if len(stress) == 3:
                    result["stress"] = stress

        return result

    def _parse_results(
        self, args: QuantumEspressoToolInput, work_dir: Path
    ) -> ToolResult:
        parsed: dict[str, Any] = {}
        for file_name in args.result_files:
            file_path = work_dir / file_name
            parsed[file_name] = self._parse_output_file(file_path)

        return ToolResult(
            data={
                "results": parsed,
                "message": f"Parsed {len(parsed)} QE output files.",
            },
            success=True,
        )
